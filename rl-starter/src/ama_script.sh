# Results of grid search
# Without Uncertainty best hyperparams....
# icm lr: 0.0001
# reward_weighting: 1.0
# novel_states_visited: 18.333333333333332
# With Uncertainty best hyperparams...
# icm lr: 0.001
# reward_weighting: 1.0
# novel_states_visited: 27.666666666666668
#rm *npy
#rm *png
#rm -r storage/*
#rm *mp4
environments=(MiniGrid-MultiRoom-N4-S5-v0)
randomise_env=False
frames=5000000
uncertainty_budget=1
random_seeds=(1 2 3 4 5)

for environment in ${environments[@]}; do
      frames_before_reset=200000000000000000
      noisy_tv=(True False)
      curiosity=(True)
      uncertainty=(True)
      save_interval=2000
      visualizing=False
      icm_lr=0.001
      reward_weighting=1.0
      normalise_rewards=True

      for random_seed in ${random_seeds[@]}; do
          for a_uncertainty in ${uncertainty[@]}; do
              for a_noisy_tv in ${noisy_tv[@]}; do
                  for a_curiosity in ${curiosity[@]}; do
                      environment_seed=$random_seed
                      python3 -m scripts.train --algo ppo --random_action False --visualizing $visualizing --normalise_rewards $normalise_rewards --env $environment --model normalise_rewards_${normalise_rewards}_frames_${frames_before_reset}_noisy_tv_${a_noisy_tv}_curiosity_${a_curiosity}_uncertainty_${a_uncertainty}_random_seed_${random_seed}_icmlr_${icm_lr}_${uncertainty_budget}_${environment}_reward_weighting_${reward_weighting} --save-interval $save_interval --frames $frames --seed $random_seed --uncertainty $a_uncertainty --noisy_tv $a_noisy_tv --curiosity $a_curiosity --randomise_env $randomise_env --environment_seed $environment_seed --icm_lr $icm_lr --reward_weighting $reward_weighting --frames_before_reset $frames_before_reset  
                  done
              done
          done
      done
      wait
done

environments=(MiniGrid-MultiRoom-N6-v0)
randomise_env=False
frames=10000000
uncertainty_budget=1
random_seeds=(1 2 3 4 5)

for environment in ${environments[@]}; do
      frames_before_reset=200000000000000000
      noisy_tv=(True False)
      curiosity=(True)
      uncertainty=(True)
      save_interval=2000
      visualizing=False
      icm_lr=0.0001
      reward_weighting=1.0
      normalise_rewards=True

      for random_seed in ${random_seeds[@]}; do
          for a_uncertainty in ${uncertainty[@]}; do
              for a_noisy_tv in ${noisy_tv[@]}; do
                  for a_curiosity in ${curiosity[@]}; do
                      environment_seed=$random_seed
                      python3 -m scripts.train --algo ppo --random_action False --visualizing $visualizing --normalise_rewards $normalise_rewards --env $environment --model normalise_rewards_${normalise_rewards}_frames_${frames_before_reset}_noisy_tv_${a_noisy_tv}_curiosity_${a_curiosity}_uncertainty_${a_uncertainty}_random_seed_${random_seed}_icmlr_${icm_lr}_${uncertainty_budget}_${environment}_reward_weighting_${reward_weighting} --save-interval $save_interval --frames $frames --seed $random_seed --uncertainty $a_uncertainty --noisy_tv $a_noisy_tv --curiosity $a_curiosity --randomise_env $randomise_env --environment_seed $environment_seed --icm_lr $icm_lr --reward_weighting $reward_weighting --frames_before_reset $frames_before_reset  
                  done
              done
          done
      done
      wait
done
