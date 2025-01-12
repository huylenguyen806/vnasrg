# Copyright 2023 Huy Le Nguyen (@nglehuy)
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

import json
import logging
import os

from tensorflow_asr import callbacks, datasets, keras, tokenizers  # import to aid logging messages
from tensorflow_asr.configs import Config
from tensorflow_asr.models.base_model import BaseModel
from tensorflow_asr.utils import cli_util, env_util, file_util, keras_util

logger = logging.getLogger(__name__)


def main(
    config_path: str,
    modeldir: str,
    dataset_type: str,
    datadir: str,
    bs: int = None,
    spx: int = 1,
    devices: list = None,
    tpu_address: str = None,
    device_type: str = "gpu",
    mxp: str = "none",
    jit_compile: bool = False,
    ga_steps: int = None,
    verbose: int = 1,
    repodir: str = os.getcwd(),
):
    keras.backend.clear_session()
    env_util.setup_seed()
    strategy = env_util.setup_strategy(device_type=device_type, devices=devices, tpu_address=tpu_address)
    env_util.setup_mxp(mxp=mxp)

    config = Config(config_path, training=True, repodir=repodir, datadir=datadir, modeldir=modeldir)

    tokenizer = tokenizers.get(config)

    train_dataset = datasets.get(
        tokenizer=tokenizer,
        dataset_config=config.data_config.train_dataset_config,
        dataset_type=dataset_type,
    )
    eval_dataset = datasets.get(
        tokenizer=tokenizer,
        dataset_config=config.data_config.eval_dataset_config,
        dataset_type=dataset_type,
    )

    model_shapes, train_batch_size, eval_batch_size, padded_shapes = datasets.get_training_shape(
        config,
        strategy,
        train_dataset,
        eval_dataset,
        batch_size=bs or config.learning_config.batch_size,
    )
    ga_steps = ga_steps or config.learning_config.ga_steps or 1

    train_data_loader = train_dataset.create(train_batch_size, ga_steps=ga_steps, padded_shapes=padded_shapes)
    logger.info(f"train_data_loader.element_spec = {json.dumps(train_data_loader.element_spec, indent=2, default=str)}")

    eval_data_loader = eval_dataset.create(eval_batch_size, padded_shapes=padded_shapes)
    if eval_data_loader:
        logger.info(f"eval_data_loader.element_spec = {json.dumps(eval_data_loader.element_spec, indent=2, default=str)}")

    with strategy.scope():
        model: BaseModel = keras_util.model_from_config(config.model_config)
        model.tokenizer = tokenizer
        output_shapes = model.make(**model_shapes)
        if config.learning_config.pretrained:
            model.load_weights(
                config.learning_config.pretrained,
                by_name=file_util.is_hdf5_filepath(config.learning_config.pretrained),
                skip_mismatch=True,
            )
        model.compile(
            optimizer=keras.optimizers.get(config.learning_config.optimizer_config),
            output_shapes=output_shapes,
            steps_per_execution=spx,
            jit_compile=jit_compile,
            ga_steps=ga_steps or config.learning_config.ga_steps,
            gwn_config=config.learning_config.gwn_config,
            gradn_config=config.learning_config.gradn_config,
        )
        model.summary()
        model.fit(
            train_data_loader,
            epochs=config.learning_config.num_epochs,
            verbose=verbose,
            validation_data=eval_data_loader,
            callbacks=callbacks.deserialize(config.learning_config.callbacks),
            steps_per_epoch=train_dataset.total_steps,
            validation_steps=eval_dataset.total_steps if eval_data_loader else None,
        )


if __name__ == "__main__":
    cli_util.run(main)