from remarc import WrightFisherEnv, Presets
from remarc.agents import train_wf_landscapes, load_testing_envs


def test_load_file():
    p = Presets.p1_test()
    train_wf_landscapes(p)
    testing_envs = load_testing_envs()
    assert isinstance(testing_envs[0], WrightFisherEnv)
