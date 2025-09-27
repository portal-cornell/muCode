from .mbpp_wrapper import MBPPWrapperEnv
from .mbpp_train_wrapper import MBPPTrainWrapperEnv
from .mbpp_plus_wrapper import MBPPPlusWrapperEnv
from .human_eval_pack_wrapper import HumanEvalPackWrapperEnv
from .human_eval_plus_wrapper import HumanEvalPlusWrapperEnv
from .code_contests_wrapper import CodeContestsWrapperEnv

from gymnasium.envs.registration import register

register(
    id='MBPPWrapperEnv-v0',
    entry_point='src.env.benchmarks.mbpp_wrapper:MBPPWrapperEnv'
)

register(
    id='MBPPTrainWrapperEnv-v0',
    entry_point='src.env.benchmarks.mbpp_train_wrapper:MBPPTrainWrapperEnv'
)

register(
    id='MBPPPlusWrapperEnv-v0',
    entry_point='src.env.benchmarks.mbpp_plus_wrapper:MBPPPlusWrapperEnv'
)

register(
    id='CodeContestsWrapperEnv-v0',
    entry_point='src.env.benchmarks.code_contests_wrapper:CodeContestsWrapperEnv'
)

register(
    id='HumanEvalPackWrapperEnv-v0',
    entry_point='src.env.benchmarks.human_eval_pack_wrapper:HumanEvalPackWrapperEnv'
)

register(
    id="HumanEvalPlusWrapperEnv-v0",
    entry_point="src.env.benchmarks.human_eval_plus_wrapper:HumanEvalPlusWrapperEnv"
)

register(
    id='LiveCodeBenchWrapperEnv-v0',
    entry_point='src.env.benchmarks.live_code_bench_wrapper:LiveCodeBenchWrapperEnv'
)