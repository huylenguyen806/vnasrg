from __future__ import absolute_import

import math
import numpy as np
import random
from featurizers.SpeechFeaturizer import read_raw_audio


def get_white_noise(signal: np.ndarray, snr=10):
  if snr == 0:
    return None
  RMS_s = math.sqrt(np.mean(signal ** 2))
  # RMS values of noise
  RMS_n = math.sqrt(RMS_s ** 2 / (pow(10, snr / 20)))
  # Additive white gausian noise. Thereore mean=0
  # Because sample length is large (typically > 40000)
  # we can use the population formula for standard daviation.
  # because mean=0 STD=RMS
  STD_n = RMS_n
  noise = np.random.normal(0, STD_n, signal.shape[0])
  return noise


def get_noise_from_sound(signal: np.ndarray, noise: np.ndarray, snr=10):
  if len(noise) < len(signal) or snr == 0:
    return None

  idx = random.choice(range(0, len(noise) - len(signal)))  # randomly crop noise wav
  noise = noise[idx:idx + len(signal)]

  RMS_s = math.sqrt(np.mean(signal ** 2))
  # required RMS of noise
  RMS_n = math.sqrt(RMS_s ** 2 / (pow(10, snr / 20)))

  # current RMS of noise
  RMS_n_current = math.sqrt(np.mean(noise ** 2))
  noise = noise * (RMS_n / (RMS_n_current + 1e-6))

  return noise


def add_noise(signal: np.ndarray, noises: list, snr_list: list,
              min_noises: int, max_noises: int, sample_rate=16000):
  random.shuffle(noises)
  num_noises = random.randint(min_noises, max_noises)
  selected_noises = random.choices(noises, k=num_noises)
  added_noises = []
  for noise_type in selected_noises:
    snr = random.choice(snr_list)
    if noise_type == "white_noise":
      noise = get_white_noise(signal, snr)
      if noise is not None:
        signal = np.add(signal, noise)
    else:
      noise = read_raw_audio(noise_type, sample_rate=sample_rate)
      noise = get_noise_from_sound(signal, noise, snr)
      if noise is not None:
        signal = np.add(signal, noise)
  return signal


def add_white_noise(signal: np.ndarray, snr_list: list):
  snr = random.choice(snr_list)
  noise = get_white_noise(signal, snr)
  if noise is not None:
    signal = np.add(signal, noise)
  return signal


def add_realworld_noise(signal: np.ndarray, noises: list, snr_list: list,
                        min_noises: int, max_noises: int, sample_rate=16000):
  random.shuffle(noises)
  num_noises = random.randint(min_noises, max_noises)
  selected_noises = random.choices(noises, k=num_noises)
  for noise_type in selected_noises:
    snr = random.choice(snr_list)
    noise = read_raw_audio(noise_type, sample_rate=sample_rate)
    noise = get_noise_from_sound(signal, noise, snr)
    if noise is not None:
      signal = np.add(signal, noise)
  return signal
