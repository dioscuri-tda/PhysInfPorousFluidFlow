TAU = 1 # lbm.cpp line 38
MU = 0.33*(TAU-0.5) # lbm.cpp line 39
FX = 2.5e-07

VELOCITY_SCALE = 1000

# max value of toruosity - if during training larger value is computed it is truncated to MAX_TORTUOSITY
# this stablize the training
MAX_TORTUOSITY = 10.0
MIN_TORTUOSITY = 0.0
