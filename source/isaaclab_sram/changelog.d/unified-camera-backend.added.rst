* Added an experimental native RGB64 PPO backend that retained both actor and critic
  weights and Adam moments in one cooperative optimizer launch for up to sixteen
  environments, with native checkpoint and random-number-state compatibility.
