import datetime
import logging
from PLM import Encoder_Decoder
from utils import set_seed

set_seed(1234)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

model_dict = {
    'codet5': 'D:/models/codet5-base',
    # 'plbart': 'D:\pretrained-model\plbart-base',
    # 'natgen': 'D:\pretrained-model\plbart-base',
}

model_type = 'codet5'
task = 'seq2seq'
datatype = 'function'

# 初始化模型
model = Encoder_Decoder(model_type=model_type, model_name_or_path=model_dict[model_type], load_model_path=None,
                        beam_size=10, max_source_length=510, max_target_length=510)

start = datetime.datetime.now()

# # 模型训练
model.train(train_filename='./train.csv', train_batch_size=8, learning_rate=5e-5,
            num_train_epochs=50, early_stop=5, do_eval=True, task=task,
            eval_filename='./valid.csv',
            eval_batch_size=8, output_dir='valid_output_' + model_type + '/',
            do_eval_bleu=True)

end = datetime.datetime.now()
print(end - start)

# 加载微调过后的模型参数
model = Encoder_Decoder(model_type=model_type, model_name_or_path=model_dict[model_type], beam_size=10,
                        max_source_length=510, max_target_length=510,
                        load_model_path='valid_output_' + model_type + '/checkpoint-best-bleu/pytorch_model.bin')

model.test(batch_size=8, filename='./test.csv',
           output_dir='test_output_' + model_type + '/')
