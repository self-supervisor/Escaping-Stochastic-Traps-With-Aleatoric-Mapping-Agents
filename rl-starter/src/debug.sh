rm *npy
rm *png
rm -r storage/*
rm *mp4
frames_before_resets=(8000000000000)
#environments tried: MiniGrid-KeyCorridorS6R3-v0, MiniGrid-DoorKey-5x5-v0, MiniGrid-DoorKey-6x6-v0, MiniGrid-DoorKey-8x8-v0
# MiniGrid-DoorKey-16x16-v0, MiniGrid-MultiRoom-N6-v0, MiniGrid-KeyCorridorS6R3-v0, MiniGrid-SimpleCrossingS11N5-v0
environment=MiniGrid-MultiRoom-N4-S5-v0 #MiniGrid-KeyCorridorS6R3-v0 #MiniGrid-MultiRoom-N4-S5-v0 #, #MiniGrid-FourRooms-v0

randomise_env=False
frames=1000000
uncertainty_budget=0.0005
#random_seeds=(85 86 87 88 89)
random_seeds=(1)
reward_weightings=(1 10 100)
normalise_rewards=(True)
icm_lrs=(0.01 0.001 0.0001)

for reward_weighting in ${reward_weightings[@]}; do
    for normalise_reward in ${normalise_rewards[@]}; do
        for icm_lr in ${icm_lrs[@]}; do
            for frames_before_reset in ${frames_before_resets[@]}; do
                random_action=False
                noisy_tv=(False)
                curiosity=(True)
                uncertainty=(True)
                save_interval=2000
                visualizing=False
               
                for random_seed in ${random_seeds[@]}; do
                    for a_uncertainty in ${uncertainty[@]}; do
                        for a_noisy_tv in ${noisy_tv[@]}; do
                            for a_curiosity in ${curiosity[@]}; do
                                environment_seed=$random_seed
                                python3 -m scripts.train --algo ppo --random_action $random_action --visualizing $visualizing --normalise_rewards $normalise_reward --env $environment --model frames_${frames_before_reset}_noisy_tv_${a_noisy_tv}_curiosity_${a_curiosity}_uncertainty_${a_uncertainty}_random_seed_${random_seed}_coefficient_${uncertainty_budget}_${environment}_icm_lr_${icm_lr}_reward_weighting_${reward_weighting}_normalise_rewards${normalise_reward} --save-interval $save_interval --frames $frames --seed $random_seed --uncertainty $a_uncertainty --noisy_tv $a_noisy_tv --curiosity $a_curiosity --randomise_env $randomise_env --environment_seed $environment_seed --icm_lr $icm_lr --reward_weighting $reward_weighting --frames_before_reset $frames_before_reset 
                            done
                        done
                    done
                done
                wait
            done
        done
    done
done
