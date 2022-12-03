# ${HOME}/dir1/user_dir/tasks.py

import functools
import seqio
import tensorflow_datasets as tfds
from t5.evaluation import metrics
from t5.data import preprocessors

vocabulary = seqio.SentencePieceVocabulary(
    'gs://t5-data/vocabs/cc_all.32000/sentencepiece.model', extra_ids=100)
output_features = {
    'inputs': seqio.Feature(vocabulary=vocabulary),
    'targets': seqio.Feature(vocabulary=vocabulary)
}

seqio.TaskRegistry.add(
    'wmt_t2t_de_en_v003',
    source=seqio.TfdsDataSource(tfds_name='wmt_t2t_translate/de-en:1.0.0'),
    preprocessors=[
        functools.partial(
            preprocessors.translate,
            source_language='de', target_language='en'),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        seqio.preprocessors.append_eos_after_trim,
    ],
    metric_fns=[metrics.bleu],
    output_features=output_features)