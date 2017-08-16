import tensorflow as tf
import argparse


def reset_tensorflow_flags():
    """
    This function is necessary because the IDE I use (Spyder) doesn't reset
    modules between every iteration, so flags will cause an error due to being
    redefined...
    
    This uses an implementation detail of tf, so may break at any tf update...
    """
    tf.flags._global_parser = argparse.ArgumentParser()