import os
import sys
import tempfile
import signal
import subprocess
import multiprocessing
import io

def subprocess_timeout_exec(code, timeout=10):
    """
    Executes code with a timeout.

    Args:
        code (str): The code (resembling a python file) to execute.
        timeout (int): The timeout in seconds.
    
    Returns:
        tuple: (str, str)
            first element is either the string representing stack trace if code raised an exception, an empty string if no exception was raised, or a string representing TimeoutError if the code timed out
            second element is either the string representing outputs to stdout or empty string if there are no logs to stdout
    """
    
    # Specify a directory you have access to, for example, your home directory
    temp_dir = os.path.expanduser('~')  # User's home directory

    with tempfile.NamedTemporaryFile(dir=temp_dir, delete=False) as temp_file:
        temp_file.write(code.encode('utf-8'))
        temp_file_path = temp_file.name

    def terminate_subprocess(proc):
        if proc.poll() is None:  # Check if the process is still running
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)  # Kill the process group
    
    proc = subprocess.Popen(
        ['python3', temp_file_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=os.setsid  # Create a new process group for the subprocess
    )

    def cleanup(signum, frame):
        terminate_subprocess(proc)
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        os._exit(1)  # Exit immediately to mimic force termination

    # Register signal handlers
    signal.signal(signal.SIGTERM, cleanup)
    signal.signal(signal.SIGINT, cleanup)

    try:
        out, errs = '', ''
        proc.wait(timeout)
        outs, errs = proc.communicate()
        outs = outs.decode('utf-8') if outs is not None else ''
        errs = errs.decode('utf-8') if errs is not None  else ''
    except subprocess.TimeoutExpired:
        errs, outs = str(TimeoutError("Code execution timed out")), ''
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        terminate_subprocess(proc)
    return errs, outs

def multiprocessing_timeout_exec(func, timeout=10, test_input=None):
    """
    Executes code with a timeout.

    Args:
        func (function): The function to execute.
        timeout (int): The timeout in seconds.
    
    Raises:
        TimeoutError: If the code execution times out.
        Exception: If the code execution raises an exception.
    
    Returns:
        str: The stdout of the code execution.
    """
    def worker(func, output_queue, test_input=None):
        if test_input is not None:
            sys.stdin = io.StringIO(test_input)
        sys.stdout = io.StringIO()
        try:
            func()
            output_queue.put(sys.stdout.getvalue())
        except Exception as e:
            exception_info = (type(e), str(e)) # Reliably reconstruct exception
            output_queue.put(exception_info)

    output_queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=worker, args=(func, output_queue, test_input))
    process.start()
    process.join(timeout)

    if process.is_alive():
        process.terminate()
        process.join()
        output_queue.close()
        output_queue.join_thread()
        raise TimeoutError("Code execution timed out")

    try:
        result = output_queue.get(timeout=timeout)
        if isinstance(result, tuple):
            exec_type, exec_message = result
            # Code execution error
            raise exec_type(exec_message)
        return result # Return stdout
    except multiprocessing.queues.Empty:
        raise Exception("Code execution failed") # If no result, code execution failed. Prevents hanging.
    finally:
        output_queue.close()
        output_queue.join_thread()

def stdin_stdout_exec(code, custom_globals, input_output_dicts):
    """
    Executes and evaluates code with custom stdin and stdout from input_output_dicts.
    
    Allows for tests that read in input from stdin and print to stdout.

    Args:
        code (string): The code to execute.
        custom_globals (dict): The globals to execute the code in.
        input_output_dicts (List[dict]): A dictionary of input and output pairs.

    Raises:
        Exception: If the code execution raises an exception.
    """
    # Redirect stdin and stdout
    def run_code(solution_code, conn, test_input):
        try:
            func = lambda: exec(solution_code, custom_globals)
            stdout_value = multiprocessing_timeout_exec(func, test_input=test_input)
            conn.send(stdout_value)
        except Exception as e:
            conn.send(e)

    executing_processes = [] # Stores (process, parent_conn, child_conn, output_data)
    for input_output_dict in input_output_dicts:
        input_data = input_output_dict["input"]
        output_data = input_output_dict["output"]
        parent_conn, child_conn = multiprocessing.Pipe()
        process = multiprocessing.Process(target=run_code, args=(code, child_conn, input_data))
        process.start()
        executing_processes.append((process, parent_conn, child_conn, output_data))
    exception_to_raise = None
    for process, parent_conn, child_conn, output_data in executing_processes:
        process.join()
        if parent_conn.poll(timeout=1):
            actual_output = parent_conn.recv()
        else:
            exception_to_raise = (Exception, "Code execution failed")
        parent_conn.close()
        child_conn.close()
        if exception_to_raise is not None:
            continue # Exception found, clean and skip rest of the processes
        if isinstance(actual_output, Exception):
            exception_to_raise = actual_output
            continue # Set first exception to error and skip rest of the processes
        output_data = output_data.strip('\n')
        actual_output = actual_output.strip('\n')
        assert actual_output == output_data, f"Expected output={output_data} for input={input_data}, but got actual_output={actual_output}"

    if exception_to_raise is not None:
        raise exception_to_raise

def functional_exec(code, custom_globals, tests):
    """
    Executes the tests on the given code.

    Args:
        code (string): The code to execute.
        custom_globals (dict): The globals to execute the code in.
        tests (List[str]): A list of test strings to run.
    
    Raises:
        Exception: If the code execution raises an exception.
    """
    def run_code(combined_code, conn):
        try:
            func = lambda: exec(combined_code, custom_globals)
            multiprocessing_timeout_exec(func)
            conn.send(None)
        except Exception as e:
            exception_info = (type(e), str(e)) # Reliably reconstruct exception
            conn.send(exception_info)
    
    executing_processes = [] # Stores (process, queue)
    for test in tests:
        combined_code = f"{code}\n{test}"
        parent_conn, child_conn = multiprocessing.Pipe()
        process = multiprocessing.Process(target=run_code, args=(combined_code, child_conn))
        process.start()
        executing_processes.append((process, parent_conn, child_conn))
    
    exception_to_raise = None
    for process, parent_conn, child_conn in executing_processes:
        process.join()
        if parent_conn.poll(timeout=1):
            result = parent_conn.recv()
        else:
            exception_to_raise = (Exception, "Code execution failed") # If no result, code execution failed. Prevents hanging.
        parent_conn.close()
        child_conn.close()
        if exception_to_raise is not None:
            continue
        if result is not None:
            exception_to_raise = result
    
    if exception_to_raise is not None:
        exec_type, exec_message = exception_to_raise
        raise exec_type(exec_message)