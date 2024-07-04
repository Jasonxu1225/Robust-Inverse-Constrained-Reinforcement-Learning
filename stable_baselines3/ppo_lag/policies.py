# This file is here just to define the TwoCriticsPolicy for PPO-Lagrangian
from stable_baselines3.common.policies import (ActorTwoCriticsPolicy,
                                               ActorTwoCriticsCnnPolicy,
                                               ActorTwoCriticsWithOpPolicy,
                                               ActorTwoCriticsWithLamPolicy,
                                               register_policy)

TwoCriticsMlpPolicy = ActorTwoCriticsPolicy

register_policy("TwoCriticsMlpPolicy", ActorTwoCriticsPolicy)
register_policy("TwoCriticsWithOpMlpPolicy", ActorTwoCriticsWithOpPolicy)
register_policy("TwoCriticsWithLamMlpPolicy", ActorTwoCriticsWithLamPolicy)
register_policy("TwoCriticsCnnPolicy", ActorTwoCriticsCnnPolicy)
