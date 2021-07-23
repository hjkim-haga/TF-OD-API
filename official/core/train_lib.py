<<<<<<< HEAD
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""TFM common training driver library."""
# pytype: disable=attribute-error
import os
from typing import Any, Mapping, Optional, Tuple

# Import libraries

from absl import logging
import orbit
import tensorflow as tf

from official.core import actions
from official.core import base_task
from official.core import base_trainer
from official.core import config_definitions
from official.core import train_utils

maybe_create_best_ckpt_exporter = train_utils.maybe_create_best_ckpt_exporter


def run_experiment(
    distribution_strategy: tf.distribute.Strategy,
    task: base_task.Task,
    mode: str,
    params: config_definitions.ExperimentConfig,
    model_dir: str,
    run_post_eval: bool = False,
    save_summary: bool = True,
    trainer: Optional[base_trainer.Trainer] = None,
    controller_cls=orbit.Controller
) -> Tuple[tf.keras.Model, Mapping[str, Any]]:
  """Runs train/eval configured by the experiment params.

  Args:
    distribution_strategy: A distribution distribution_strategy.
    task: A Task instance.
    mode: A 'str', specifying the mode. Can be 'train', 'eval', 'train_and_eval'
      or 'continuous_eval'.
    params: ExperimentConfig instance.
    model_dir: A 'str', a path to store model checkpoints and summaries.
    run_post_eval: Whether to run post eval once after training, metrics logs
      are returned.
    save_summary: Whether to save train and validation summary.
    trainer: the base_trainer.Trainer instance. It should be created within the
      strategy.scope().
    controller_cls: The controller class to manage the train and eval process.
      Must be a orbit.Controller subclass.

  Returns:
    A 2-tuple of (model, eval_logs).
      model: `tf.keras.Model` instance.
      eval_logs: returns eval metrics logs when run_post_eval is set to True,
        otherwise, returns {}.
  """

  with distribution_strategy.scope():
    if not trainer:
      trainer = train_utils.create_trainer(
          params,
          task,
          train='train' in mode,
          evaluate=('eval' in mode) or run_post_eval,
          checkpoint_exporter=maybe_create_best_ckpt_exporter(
              params, model_dir))

  if trainer.checkpoint:
    if model_dir is None:
      raise ValueError('model_dir must be specified, but got None')
    checkpoint_manager = tf.train.CheckpointManager(
        trainer.checkpoint,
        directory=model_dir,
        max_to_keep=params.trainer.max_to_keep,
        step_counter=trainer.global_step,
        checkpoint_interval=params.trainer.checkpoint_interval,
        init_fn=trainer.initialize)
    # Adds recovery handling.
    trainer.add_recovery(params.trainer, checkpoint_manager=checkpoint_manager)
  else:
    checkpoint_manager = None

  controller = controller_cls(
      strategy=distribution_strategy,
      trainer=trainer if 'train' in mode else None,
      evaluator=trainer,
      global_step=trainer.global_step,
      steps_per_loop=params.trainer.steps_per_loop,
      checkpoint_manager=checkpoint_manager,
      summary_dir=os.path.join(model_dir, 'train') if (save_summary) else None,
      eval_summary_dir=os.path.join(model_dir,
                                    params.trainer.validation_summary_subdir) if
      (save_summary) else None,
      summary_interval=params.trainer.summary_interval if
      (save_summary) else None,
      eval_actions=actions.get_eval_actions(params, trainer, model_dir))

  logging.info('Starts to execute mode: %s', mode)
  with distribution_strategy.scope():
    if mode == 'train':
      controller.train(steps=params.trainer.train_steps)
    elif mode == 'train_and_eval':
      controller.train_and_evaluate(
          train_steps=params.trainer.train_steps,
          eval_steps=params.trainer.validation_steps,
          eval_interval=params.trainer.validation_interval)
    elif mode == 'eval':
      controller.evaluate(steps=params.trainer.validation_steps)
    elif mode == 'continuous_eval':

      def timeout_fn():
        if trainer.global_step.numpy() >= params.trainer.train_steps:
          return True
        return False

      controller.evaluate_continuously(
          steps=params.trainer.validation_steps,
          timeout=params.trainer.continuous_eval_timeout,
          timeout_fn=timeout_fn)
    else:
      raise NotImplementedError('The mode is not implemented: %s' % mode)

  num_params = train_utils.try_count_params(trainer.model)
  if num_params is not None:
    logging.info('Number of trainable params in model: %f Millions.',
                 num_params / 10.**6)

  if run_post_eval:
    with distribution_strategy.scope():
      return trainer.model, trainer.evaluate(
          tf.convert_to_tensor(params.trainer.validation_steps))
  else:
    return trainer.model, {}
=======
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""TFM common training driver library."""
# pytype: disable=attribute-error
import os
from typing import Any, Mapping, Optional, Tuple

# Import libraries

from absl import logging
import orbit
import tensorflow as tf

from official.core import actions
from official.core import base_task
from official.core import base_trainer
from official.core import config_definitions
from official.core import train_utils

maybe_create_best_ckpt_exporter = train_utils.maybe_create_best_ckpt_exporter


def run_experiment(
    distribution_strategy: tf.distribute.Strategy,
    task: base_task.Task,
    mode: str,
    params: config_definitions.ExperimentConfig,
    model_dir: str,
    run_post_eval: bool = False,
    save_summary: bool = True,
    trainer: Optional[base_trainer.Trainer] = None,
    controller_cls=orbit.Controller
) -> Tuple[tf.keras.Model, Mapping[str, Any]]:
  """Runs train/eval configured by the experiment params.

  Args:
    distribution_strategy: A distribution distribution_strategy.
    task: A Task instance.
    mode: A 'str', specifying the mode. Can be 'train', 'eval', 'train_and_eval'
      or 'continuous_eval'.
    params: ExperimentConfig instance.
    model_dir: A 'str', a path to store model checkpoints and summaries.
    run_post_eval: Whether to run post eval once after training, metrics logs
      are returned.
    save_summary: Whether to save train and validation summary.
    trainer: the base_trainer.Trainer instance. It should be created within the
      strategy.scope().
    controller_cls: The controller class to manage the train and eval process.
      Must be a orbit.Controller subclass.

  Returns:
    A 2-tuple of (model, eval_logs).
      model: `tf.keras.Model` instance.
      eval_logs: returns eval metrics logs when run_post_eval is set to True,
        otherwise, returns {}.
  """

  with distribution_strategy.scope():
    if not trainer:
      trainer = train_utils.create_trainer(
          params,
          task,
          train='train' in mode,
          evaluate=('eval' in mode) or run_post_eval,
          checkpoint_exporter=maybe_create_best_ckpt_exporter(
              params, model_dir))

  if trainer.checkpoint:
    if model_dir is None:
      raise ValueError('model_dir must be specified, but got None')
    checkpoint_manager = tf.train.CheckpointManager(
        trainer.checkpoint,
        directory=model_dir,
        max_to_keep=params.trainer.max_to_keep,
        step_counter=trainer.global_step,
        checkpoint_interval=params.trainer.checkpoint_interval,
        init_fn=trainer.initialize)
    # Adds recovery handling.
    trainer.add_recovery(params.trainer, checkpoint_manager=checkpoint_manager)
  else:
    checkpoint_manager = None

  controller = controller_cls(
      strategy=distribution_strategy,
      trainer=trainer if 'train' in mode else None,
      evaluator=trainer,
      global_step=trainer.global_step,
      steps_per_loop=params.trainer.steps_per_loop,
      checkpoint_manager=checkpoint_manager,
      summary_dir=os.path.join(model_dir, 'train') if (save_summary) else None,
      eval_summary_dir=os.path.join(model_dir,
                                    params.trainer.validation_summary_subdir) if
      (save_summary) else None,
      summary_interval=params.trainer.summary_interval if
      (save_summary) else None,
      train_actions=actions.get_train_actions(params, trainer, model_dir),
      eval_actions=actions.get_eval_actions(params, trainer, model_dir))

  logging.info('Starts to execute mode: %s', mode)
  with distribution_strategy.scope():
    if mode == 'train':
      controller.train(steps=params.trainer.train_steps)
    elif mode == 'train_and_eval':
      controller.train_and_evaluate(
          train_steps=params.trainer.train_steps,
          eval_steps=params.trainer.validation_steps,
          eval_interval=params.trainer.validation_interval)
    elif mode == 'eval':
      controller.evaluate(steps=params.trainer.validation_steps)
    elif mode == 'continuous_eval':

      def timeout_fn():
        if trainer.global_step.numpy() >= params.trainer.train_steps:
          return True
        return False

      controller.evaluate_continuously(
          steps=params.trainer.validation_steps,
          timeout=params.trainer.continuous_eval_timeout,
          timeout_fn=timeout_fn)
    else:
      raise NotImplementedError('The mode is not implemented: %s' % mode)

  num_params = train_utils.try_count_params(trainer.model)
  if num_params is not None:
    logging.info('Number of trainable params in model: %f Millions.',
                 num_params / 10.**6)

  flops = train_utils.try_count_flops(trainer.model)
  if flops is not None:
    logging.info('FLOPs (multi-adds) in model: %f Billions.',
                 flops / 10.**9 / 2)

  if run_post_eval:
    with distribution_strategy.scope():
      return trainer.model, trainer.evaluate(
          tf.convert_to_tensor(params.trainer.validation_steps))
  else:
    return trainer.model, {}
>>>>>>> 0650ea24129892fb026a27b37028b500fb9383fa
