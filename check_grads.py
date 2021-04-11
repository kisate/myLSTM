import numpy as np
from functions import *
from model import *
from grad_checker import *
from grad_checks import *
np.random.seed(100500)

run_checks(check_softmax_ce, 10, 1e-7, "softmax_ce")
run_checks(check_one_layer, 10, 1e-7, "One to One")
run_checks(check_many_to_one, 10, 1e-7, "Many to One")
run_checks(check_many_to_many, 10, 1e-7, "Many to Many")
run_checks(check_one_to_one_params, 10, 1e-7, "OtO params")
run_checks(check_many_to_many_params, 1, 1e-7, "MtM params")
run_checks(check_encoder, 10, 1e-7, "Encoder")
run_checks(check_encoder_params, 10, 1e-7, "Encoder params")
run_checks(check_decoder, 4, 1e-7, "Decoder params")
run_checks(check_softmax, 4, 1e-7, "softmax")
run_checks(check_decoder, 10, 1e-7, "Decoder")
run_checks(check_decoder_with_encoder, 10, 1e-7, "Decoder with encoder")
