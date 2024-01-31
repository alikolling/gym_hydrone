from gym_hydrone.utils import lazy_loader

with lazy_loader.LazyImports(__name__, False):
    from gym_hydrone.wrappers.gym_wrapper import GymWrapper
    from gym_hydrone.wrappers.gymnasium_wrapper import GymnasiumWrapper

del lazy_loader
