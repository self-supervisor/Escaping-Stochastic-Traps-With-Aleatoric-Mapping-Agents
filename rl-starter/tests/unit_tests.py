import pytest
from ..src.scripts.a2c import A2C
from ..src.scripts.models import AutoencoderWithUncertainty
from ..src.model import ACModel

@pytest.fixture
def a2c_algo():
    device = "cpu"
    icm_lr = 
    autoencoder = AutoencoderWithUncertainty(observation_shape=(7, 7, 3)).to(device) 
    autoencoder_opt = torch.optim.Adam(
        autoencoder.parameters(), lr=icm_lr, weight_decay=0
    )
    uncertainty = True
    noisy_tv = True
    curiosity = True

    algo = A2CAlgo(                                                                                                                           
            envs,                                                                                                                                 
            acmodel,                                                                                                                              
            autoencoder,                                                                                                                          
            autoencoder_opt,                                                                                                                      
            uncertainty,                                                                                                                     
            noisy_tv,                                                                                                                        
            curiosity,                                                                                                                       
            randomise_env,                                                                                                                   
            uncertainty_budget,                                                                                                              
            environment_seed,                                                                                                                
            reward_weighting,                                                                                                                     
            normalise_rewards,                                                                                                                    
            args.frames_before_reset,                                                                                                             
            device,                                                                                                                               
            args.frames_per_proc,                                                                                                                 
            args.discount,                                                                                                                        
            args.lr,                                                                                                                              
            args.gae_lambda,                                                                                                                      
            args.entropy_coef,                                                                                                                    
            args.value_loss_coef,                                                                                                                 
            args.max_grad_norm,                                                                                                                   
            args.recurrence,                                                                                                                      
            args.optim_alpha,                                                                                                                     
            args.optim_eps,                                                                                                                       
            preprocess_obss)
    return algo


@pytest.mark.parametrize()
def test_update_visitation_counts():
    pass


@pytest.mark.parametrize()
def test_add_noisy_tv():
    pass


@pytest.mark.parametrize()
def test_reset_environments_if_ness():
    pass


@pytest.mark.parametrize()
def test_compute_intrinsic_rewards():
    pass


def test_get_mean_and_std_dev():
    pass


def test_get_label_from_path():
    pass


def test_plot():
    pass
