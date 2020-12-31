rm *npy
rm *png
rm -r storage/*
rm *mp4
frames_before_resets=(8)
#environments tried: MiniGrid-KeyCorridorS6R3-v0, MiniGrid-DoorKey-5x5-v0, MiniGrid-DoorKey-6x6-v0, MiniGrid-DoorKey-8x8-v0
# MiniGrid-DoorKey-16x16-v0, MiniGrid-MultiRoom-N6-v0, MiniGrid-KeyCorridorS6R3-v0, MiniGrid-SimpleCrossingS11N5-v0
environment=MiniGrid-KeyCorridorS6R3-v0 #, #MiniGrid-FourRooms-v0

randomise_env=False
frames=500
uncertainty_budget=0.0005
random_seeds=(85 86 87 88 89)
random_seeds=(1 2 3)

for frames_before_reset in ${frames_before_resets[@]}; do

    
    random_action=True
    reward_weighting=10
    noisy_tv=(True False)
    curiosity=(True)
    uncertainty=(True)
    save_interval=2000
    normalise_rewards=False
    icm_lr=0.001
    visualizing=False
   
    for random_seed in ${random_seeds[@]}; do
        for a_uncertainty in ${uncertainty[@]}; do
            for a_noisy_tv in ${noisy_tv[@]}; do
                for a_curiosity in ${curiosity[@]}; do
                    environment_seed=$random_seed
                    python3 -m scripts.train --algo a2c --random_action True --visualizing $visualizing --normalise_rewards True --env $environment --model frames_${frames_before_reset}_noisy_tv_${a_noisy_tv}_random_action_${random_action}_random_seed_${random_seed}_coefficient_${uncertainty_budget}_${environment} --save-interval $save_interval --frames $frames --seed $random_seed --uncertainty $a_uncertainty --noisy_tv $a_noisy_tv --curiosity $a_curiosity --randomise_env $randomise_env --environment_seed $environment_seed --icm_lr $icm_lr --reward_weighting $reward_weighting --frames_before_reset $frames_before_reset & 
                done
            done
        done
    done
    wait
   
    random_action=False
    reward_weighting=10
    noisy_tv=(True False)
    curiosity=(True)
    uncertainty=(True)
    save_interval=2000
    normalise_rewards=False
    icm_lr=0.0001
    visualizing=False
   
    for random_seed in ${random_seeds[@]}; do
        for a_uncertainty in ${uncertainty[@]}; do
            for a_noisy_tv in ${noisy_tv[@]}; do
                for a_curiosity in ${curiosity[@]}; do
                    environment_seed=$random_seed
                    python3 -m scripts.train --algo a2c --random_action False --visualizing $visualizing --normalise_rewards True --env $environment --model frames_${frames_before_reset}_noisy_tv_${a_noisy_tv}_curiosity_${a_curiosity}_uncertainty_${a_uncertainty}_random_seed_${random_seed}_coefficient_${uncertainty_budget}_${environment} --save-interval $save_interval --frames $frames --seed $random_seed --uncertainty $a_uncertainty --noisy_tv $a_noisy_tv --curiosity $a_curiosity --randomise_env $randomise_env --environment_seed $environment_seed --icm_lr $icm_lr --reward_weighting $reward_weighting --frames_before_reset $frames_before_reset & 
                done
            done
        done
    done
    wait

    reward_weighting=10
    icm_lr=0.001
    noisy_tv=(True False)
    curiosity=(True)
    uncertainty=(False)
    normalise_reward=False

    for random_seed in ${random_seeds[@]}; do
        for a_uncertainty in ${uncertainty[@]}; do
            for a_noisy_tv in ${noisy_tv[@]}; do
                for a_curiosity in ${curiosity[@]}; do
                    environment_seed=$random_seed
                    python3 -m scripts.train --algo a2c --visualizing $visualizing --random_action False --normalise_rewards $normalise_reward --env $environment --model frames_${frames_before_reset}_noisy_tv_${a_noisy_tv}_curiosity_${a_curiosity}_uncertainty_${a_uncertainty}_random_seed_${random_seed}_coefficient_${uncertainty_budget}_${environment} --icm_lr $icm_lr --reward_weighting $reward_weighting --save-interval $save_interval --frames $frames --seed $random_seed --uncertainty $a_uncertainty --noisy_tv $a_noisy_tv --curiosity $a_curiosity --randomise_env $randomise_env --environment_seed $environment_seed --frames_before_reset $frames_before_reset & 
                done
            done
        done
    done
    wait

    reward_weighting=10
    icm_lr=0.0001
    noisy_tv=(True False)
    curiosity=(False)
    uncertainty=(False)
    normalise_reward=False

    for random_seed in ${random_seeds[@]}; do
        for a_uncertainty in ${uncertainty[@]}; do
            for a_noisy_tv in ${noisy_tv[@]}; do
                for a_curiosity in ${curiosity[@]}; do
                    environment_seed=$random_seed
                    python3 -m scripts.train --algo a2c --visualizing $visualizing --random_action False --normalise_rewards $normalise_reward --env $environment --model frames_${frames_before_reset}_noisy_tv_${a_noisy_tv}_curiosity_${a_curiosity}_uncertainty_${a_uncertainty}_random_seed_${random_seed}_coefficient_${uncertainty_budget} --icm_lr $icm_lr --reward_weighting $reward_weighting --save-interval $save_interval --frames $frames --seed $random_seed --uncertainty $a_uncertainty --noisy_tv $a_noisy_tv --curiosity $a_curiosity --randomise_env $randomise_env --environment_seed $environment_seed --frames_before_reset $frames_before_reset & 
                done
            done
        done
    done
    wait
    python3 -m scripts.plot --environment $environment
done
