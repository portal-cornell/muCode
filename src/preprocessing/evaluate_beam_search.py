import argparse
import os
from datasets import Dataset

def process_rollouts(datasets):
    rollouts = []
    dataset_len = len(datasets[0])
    for i in range(dataset_len):
        rollout = []
        prev_prompt = None
        prev_data_id = None
        for j in range(len(datasets)):
            data = datasets[j][i]
            # Check correctness and set variables
            assert len(data['prompt']) == 2 * j + 1 # Prompt size increases from 1, 3, 5, ...
            curr_prompt = data['prompt']
            assert len(data['completion']) == 1
            curr_completion = data['completion'][0]
            assert 'data_id' in data
            curr_data_id = data['data_id']
            assert len(data['feedbacks']) == 1 # Should only be taking one step at a time
            feedbacks = data['feedbacks'][0]

            # Check consistency and set previous variables
            assert prev_prompt is None or prev_prompt == curr_prompt[:len(prev_prompt)]
            assert prev_data_id is None or prev_data_id == curr_data_id
            # if not feedbacks['public']:
            #     import pdb; pdb.set_trace()
            # assert feedbacks['public'] # Must not be None or empty string
            prev_prompt = curr_prompt
            prev_data_id = curr_data_id
            passed_public_tests = feedbacks['public'] == "Code passed all tests"
            passed_private_tests = feedbacks['private'] in ["Code passed all tests", ""]
            rollout_step = (curr_prompt, curr_completion, curr_data_id, passed_public_tests, passed_private_tests)
            rollout.append(rollout_step)
        rollouts.append(rollout)
    return rollouts
            
        
def calculate_performance(data_paths, output_path):
    datasets = [Dataset.load_from_disk(data_path) for data_path in data_paths]
    assert len(datasets) > 0
    rollouts = process_rollouts(datasets)
    assert all([len(rollout) == len(datasets) for rollout in rollouts])
    turn_success = [0] * len(datasets)
    turn_pub_success = [0] * len(datasets)


    problem_dict = {}
    for rollout in rollouts:
        data_id = rollout[0][2]
        problem_dict[data_id] = problem_dict.get(data_id, [0] * len(datasets)) # Initialize to 0 if not found
        for turn, rollout_step in enumerate(rollout):
            _, _, _, passed_public_tests, passed_private_tests = rollout_step
            if passed_public_tests:
                turn_success = [success + passed_private_tests if i >= turn else success for i, success in enumerate(turn_success)]
                turn_pub_success = [success + 1 if i >= turn else success for i, success in enumerate(turn_pub_success)]
                problem_dict[data_id] = [success or passed_private_tests if i >= turn else success for i, success in enumerate(problem_dict[data_id])]
                break
                
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output_dir = os.path.dirname(output_path)
    output_basename = os.path.basename(output_path)

    print(f"=======   True success   =======")
    for i, success in enumerate(turn_success):
        true_success = success/len(datasets[0])
        print(f"Turn {i+1}: {true_success}")
        with open(output_path, "a") as f:
            f.write(f"Turn {i+1}: {true_success}\n")
    
    print(f"=======   Public test success   =======")
    public_output_path = os.path.join(output_dir, f"public_{output_basename}")
    for i, success in enumerate(turn_pub_success):
        public_test_success = success/len(datasets[0])
        print(f"Turn {i+1}: {public_test_success}")
        with open(public_output_path, "a") as f:
            f.write(f"Turn {i+1}: {public_test_success}\n")
    
    print(f"=======   Any success   =======")
    any_output_path = os.path.join(output_dir, f"any_{output_basename}")
    for i in range(len(datasets)):
        any_success = sum([problem[i] for problem in problem_dict.values()])/len(problem_dict)
        print(f"Turn {i+1}: {any_success}")
        with open(any_output_path, "a") as f:
            f.write(f"Turn {i+1}: {any_success}\n")

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--top_dataset_paths", type=str, required=True, help="The paths to each of the top K rollouts for beam search.")
    args.add_argument("--output_path", type=str, required=True, help="The path to save the results.")
    args = args.parse_args()

    dataset_paths = args.top_dataset_paths.split(",")
    calculate_performance(dataset_paths, args.output_path)