rm *npy
rm *png
rm -r storage/*
rm *mp4
frames_before_resets=(2000000)
#environments tried: MiniGrid-KeyCorridorS6R3-v0, MiniGrid-DoorKey-5x5-v0, MiniGrid-DoorKey-6x6-v0, MiniGrid-DoorKey-8x8-v0
# MiniGrid-DoorKey-16x16-v0, MiniGrid-MultiRoom-N6-v0, MiniGrid-KeyCorridorS6R3-v0, MiniGrid-SimpleCrossingS11N5-v0
environment=MiniGrid-MultiRoom-N4-S5-v0 #MiniGrid-FourRooms-v0

randomise_env=False
frames=2000000
uncertainty_budget=1 #0.0005
#random_seeds=(85 86 87 88 89)
random_seeds=(3)
reward_weightings=(10)
icm_lrs=(0.001)
normalise_rewards_list=(True) #True
random_action=False

for normalise_rewards in ${normalise_rewards_list[@]}; do
    for reward_weighting in ${reward_weightings[@]}; do 
      for icm_lr in ${icm_lrs[@]}; do 
          for frames_before_reset in ${frames_before_resets[@]}; do
              noisy_tv=(False)
              curiosity=(True)
              uncertainty=(False)
              save_interval=2000
              visualizing=False
             
              for random_seed in ${random_seeds[@]}; do
                  for a_uncertainty in ${uncertainty[@]}; do
                      for a_noisy_tv in ${noisy_tv[@]}; do
                          for a_curiosity in ${curiosity[@]}; do
                              environment_seed=$random_seed
                              rm *npy
                              rm *png
                              rm -r storage/*
                              rm *mp4
                              python3 -m scripts.train --algo a2c --random_action $random_action --visualizing $visualizing --normalise_rewards $normalise_rewards --env $environment --model frames_${frames_before_reset}_noisy_tv_${a_noisy_tv}_curiosity_${a_curiosity}_uncertainty_${a_uncertainty}_random_seed_${random_seed}_coefficient_${uncertainty_budget}_${environment} --save-interval $save_interval --frames $frames --seed $random_seed --uncertainty $a_uncertainty --noisy_tv $a_noisy_tv --curiosity $a_curiosity --randomise_env $randomise_env --environment_seed $environment_seed --icm_lr $icm_lr --reward_weighting $reward_weighting --frames_before_reset $frames_before_reset 
                          done
                      done
                  done
              done
            done 
        done
    done
done
