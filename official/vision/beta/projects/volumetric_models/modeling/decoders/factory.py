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

# Lint as: python3
"""factory method."""

from typing import Mapping
import tensorflow as tf

from official.vision.beta.projects.volumetric_models.modeling import decoders


def build_decoder(
    input_specs: Mapping[str, tf.TensorShape],
    model_config,
    l2_regularizer: tf.keras.regularizers.Regularizer = None) -> tf.keras.Model:
  """Builds decoder from a config.

  Args:
    input_specs: `dict` input specifications. A dictionary consists of
      {level: TensorShape} from a backbone.
    model_config: A OneOfConfig. Model config.
    l2_regularizer: tf.keras.regularizers.Regularizer instance. Default to None.

  Returns:
    A tf.keras.Model instance of the decoder.
  """
  decoder_type = model_config.decoder.type
  decoder_cfg = model_config.decoder.get()
  norm_activation_config = model_config.norm_activation

  if decoder_type == 'identity':
    decoder = None
  elif decoder_type == 'unet_3d_decoder':
    decoder = decoders.UNet3DDecoder(
        model_id=decoder_cfg.model_id,
        input_specs=input_specs,
        pool_size=decoder_cfg.pool_size,
        kernel_regularizer=l2_regularizer,
        activation=norm_activation_config.activation,
        norm_momentum=norm_activation_config.norm_momentum,
        norm_epsilon=norm_activation_config.norm_epsilon,
        use_sync_bn=norm_activation_config.use_sync_bn,
        use_batch_normalization=decoder_cfg.use_batch_normalization,
        use_deconvolution=decoder_cfg.use_deconvolution)
  else:
    raise ValueError('Decoder {!r} not implement'.format(decoder_type))

  return decoder
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

"""Decoder registers and factory method.

One can register a new decoder model by the following two steps:

1 Import the factory and register the build in the decoder file.
2 Import the decoder class and add a build in __init__.py.

```
# my_decoder.py

from modeling.decoders import factory

class MyDecoder():
  ...

@factory.register_decoder_builder('my_decoder')
def build_my_decoder():
  return MyDecoder()

# decoders/__init__.py adds import
from modeling.decoders.my_decoder import MyDecoder
```

If one wants the MyDecoder class to be used only by those binary
then don't imported the decoder module in decoders/__init__.py, but import it
in place that uses it.
"""
from typing import Union, Mapping, Optional

# Import libraries

import tensorflow as tf

from official.core import registry
from official.modeling import hyperparams

_REGISTERED_DECODER_CLS = {}


def register_decoder_builder(key: str):
  """Decorates a builder of decoder class.

  The builder should be a Callable (a class or a function).
  This decorator supports registration of decoder builder as follows:

  ```
  class MyDecoder(tf.keras.Model):
    pass

  @register_decoder_builder('mydecoder')
  def builder(input_specs, config, l2_reg):
    return MyDecoder(...)

  # Builds a MyDecoder object.
  my_decoder = build_decoder_3d(input_specs, config, l2_reg)
  ```

  Args:
    key: A `str` of key to look up the builder.

  Returns:
    A callable for using as class decorator that registers the decorated class
    for creation from an instance of task_config_cls.
  """
  return registry.register(_REGISTERED_DECODER_CLS, key)


@register_decoder_builder('identity')
def build_identity(
    input_specs: Optional[Mapping[str, tf.TensorShape]] = None,
    model_config: Optional[hyperparams.Config] = None,
    l2_regularizer: Optional[tf.keras.regularizers.Regularizer] = None) -> None:
  del input_specs, model_config, l2_regularizer  # Unused by identity decoder.
  return None


def build_decoder(
    input_specs: Mapping[str, tf.TensorShape],
    model_config: hyperparams.Config,
    l2_regularizer: tf.keras.regularizers.Regularizer = None,
    **kwargs) -> Union[None, tf.keras.Model, tf.keras.layers.Layer]:
  """Builds decoder from a config.

  Args:
    input_specs: A `dict` of input specifications. A dictionary consists of
      {level: TensorShape} from a backbone.
    model_config: A `OneOfConfig` of model config.
    l2_regularizer: A `tf.keras.regularizers.Regularizer` object. Default to
      None.
    **kwargs: Additional keyword args to be passed to decoder builder.

  Returns:
    An instance of the decoder.
  """
  decoder_builder = registry.lookup(_REGISTERED_DECODER_CLS,
                                    model_config.decoder.type)

  return decoder_builder(
      input_specs=input_specs,
      model_config=model_config,
      l2_regularizer=l2_regularizer,
      **kwargs)
>>>>>>> 0650ea24129892fb026a27b37028b500fb9383fa
