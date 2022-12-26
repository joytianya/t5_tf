# coding: utf-8

import t5
import seqio
import functools
import tensorflow.compat.v1 as tf
# import tensorflow_datasets as tfds
# from sklearn.metrics import f1_score, precision_score, recall_score
from t5.evaluation import metrics
import random
import json
import logging
logging.basicConfig(level=logging.INFO)
# 定义sentencepiece模型的位置
# 这里我是用的是mT5模型的词表模型，如果使用T5模型的话，可以改为gs://t5-data/vocabs/cc_all.32000/sentencepiece.model
# DEFAULT_SPM_PATH = "gs://t5-data/vocabs/mc4.250000.100extra/sentencepiece.model" # THIS IS FOR MT5 ONLY
# DEFAULT_SPM_PATH = "gs://clueai/models/xl/vocab/spiece.model"
#DEFAULT_SPM_PATH = "gs://clueai/data/brightdata/m0905_32128.model" # m0905_v0.model"
#DEFAULT_SPM_PATH = "gs://clueai/data/zxw/tf/spm/lanzhou/spiece.model"
#DEFAULT_SPM_PATH = "gs://clueai/data/zxw/tf/spm/clueai/spm_clueai_0920.model"
DEFAULT_SPM_PATH = "gs://t5-data/vocabs/mc4.250000.100extra/sentencepiece.model"
DEFAULT_VOCAB = t5.data.SentencePieceVocabulary(DEFAULT_SPM_PATH)

DEFAULT_OUTPUT_FEATURES = {
	"inputs": t5.data.Feature(vocabulary=DEFAULT_VOCAB, add_eos=True, required=False),
	"targets": t5.data.Feature(vocabulary=DEFAULT_VOCAB, add_eos=True)
}

# 自定义的数据集生成函数
# 这个函数必须有split和shuffle_files这两个positional parameters(位置不能乱)
# 后面那个lang是因为我的数据集是多语言的，我因为懒的写太多functions，所以传了一个closure进来
def customize_finetuning_dataset_fn(split, shuffle_files=False, seed=None, dataset_name="*", glob_num_str="*", lang="multilingual"):
	dataset_type = "*"
	logging.info(f"dataset_type: {dataset_type}, dataset_name: {dataset_name}, glob_num_str: {glob_num_str}")
	if split == "dev":
		glob_num_str = "[0][0][0][0][0][0]"
	csvdata = f"gs://clueai/data/zxw/tf/data/csvdata_replace_huanhang_multi_files_chat/{dataset_type}/{dataset_name}/*{split}-{glob_num_str}.csv" 
	logging.info(f"csvdata: {csvdata}")
	files_dataset = tf.data.TextLineDataset.list_files(csvdata, shuffle=True)
	ds = tf.data.TextLineDataset(files_dataset)
	ds = ds.map(functools.partial(tf.io.decode_csv,
								  record_defaults=["", ""],
								  field_delim=",", # TODO 0910. use default delim-->field_delim="&",#""\t",
								  use_quote_delim=True),
				num_parallel_calls=tf.data.experimental.AUTOTUNE)
	#ds = ds.shuffle(100000, reshuffle_each_iteration=True)
	ds = ds.map(lambda *ex: dict(zip(["input", "output"], ex)), num_parallel_calls=tf.data.experimental.AUTOTUNE)
	# 缓存实验试试
	#ds = ds.cache()
	return ds

def customize_pretrain_dataset_fn_mt(split, shuffle_files=False, seed=None, lang="multilingual"):
	"""
	设置多任务微调(fine-tuning)的数据源及其处理
	:param split:
	:param shuffle_files:
	:param lang:
	:return:
	"""
	
	#glob_num_str = "[0][0][0][0-5][0-9][0-9][0-9]"
	glob_num_str = "*"
	# 在训练集合上进行继续预训练
	textdata = f"gs://clueai/data/zxw/tf/data/pretrain_data_replace_huanhang_multi_files_chat/*/*/*_train-{glob_num_str}.txt"
	files_dataset = tf.data.TextLineDataset.list_files(textdata, shuffle=True)
	# 总共文件个数：33773
	all_files_num = tf.data.Dataset.cardinality(files_dataset)
	#print(f"总共数据集个数: {all_files_num}")
	logging.info(f"总共数据集个数: {all_files_num}")
	# 采样的个数
	#take_num = all_files_num//6
	take_num = all_files_num
	files_dataset = files_dataset.take(take_num)
	logging.info(f"采样的个数: {take_num}")
	#files_dataset = files_dataset.shuffle(1000, reshuffle_each_iteration=True)
	ds = tf.data.TextLineDataset(files_dataset)
	#ds = ds.shuffle(10000, reshuffle_each_iteration=True)
	
	return ds

def customize_pretrain_dataset_fn(split, shuffle_files=False, seed=None, lang="multilingual"):
	"""
	设置多任务微调(fine-tuning)的数据源及其处理
	:param split:
	:param shuffle_files:
	:param lang:
	:return:
	"""
	
	#glob_num_str = "[0][0][0][0-5][0-9][0-9][0-9]"
	glob_num_str = "*"
	textdata = f"gs://clueai/data/zxw/tf/data/pretrain_data/*/*_{glob_num_str}.txt"
	# 在训练集合上进行继续预训练
	#textdata = f"gs://clueai/data/zxw/tf/data/pretrain_data_replace_huanhang_multi_files/*/*/*_train-{glob_num_str}.txt"
	files_dataset = tf.data.TextLineDataset.list_files(textdata, shuffle=True)
	# 总共文件个数：33773
	all_files_num = tf.data.Dataset.cardinality(files_dataset)
	#print(f"总共数据集个数: {all_files_num}")
	logging.info(f"总共数据集个数: {all_files_num}")
	# 采样的个数
	take_num = all_files_num//10
	files_dataset = files_dataset.take(take_num)
	logging.info(f"采样的个数: {take_num}")
	#files_dataset = files_dataset.shuffle(1000, reshuffle_each_iteration=True)
	ds = tf.data.TextLineDataset(files_dataset)
	#ds = ds.shuffle(10000, reshuffle_each_iteration=True)
	
	return ds
	
# 自定义的预处理函数
# 这里可以定义把你的自定义数据送进模型之前的预处理函数
def customize_finetuning_preprocessor(ds):
	def to_inputs_and_targets(ex):
		return {
			"inputs": ex["input"],
			"targets": ex["output"]
		}
	return ds.map(to_inputs_and_targets, num_parallel_calls=tf.data.experimental.AUTOTUNE)

def customize_pretrain_preprocessor(ds):
	def to_inputs_and_targets(ex):
		return {
			"targets": ex
		}
	return ds.map(to_inputs_and_targets, num_parallel_calls=tf.data.experimental.AUTOTUNE)

def customize_finetuning_preprocessor_decoder(ds):
	def to_inputs_and_targets(ex):
		return {
			"inputs": "",
			"targets": ex["output"]
		}
	return ds.map(to_inputs_and_targets, num_parallel_calls=tf.data.experimental.AUTOTUNE)


def customize_pretrain_preprocessor_encoder(ds):
	def to_inputs_and_targets(ex):
		return {
			"targets": ex["input"]
		}
	return ds.map(to_inputs_and_targets, num_parallel_calls=tf.data.experimental.AUTOTUNE)

def customize_pretrain_preprocessor_encoder_decoder(ds):
	def to_inputs_and_targets(ex):
		return {
			"targets": ex["input"] + ex["output"]
		}
	return ds.map(to_inputs_and_targets, num_parallel_calls=tf.data.experimental.AUTOTUNE)

# Task 定义 321个数据集
# 针对每一个细分的dataset，我们可以定义一个task，注意这里使用的是seqio的TaskRegistry，不是T5.data.TaskRegistry
#task_weights_321 = {'kgclue': 19731218, 'translation2019zhen': 5131434, 'translation2019enzh': 5131434, 'wmt_20_zh_en': 5000002, 'wmt_20_en_zh': 5000002, 'wudao_article_generation': 4255462, 'web_text': 4111080, 'yf_amazon': 3774069, 'PyTorrent_code_generation_docstring': 2258188, 'product_desc_ali': 2109187, 'cnki_article_generation': 1966982, 'cnki_summary': 1966982, 'yf_dianping': 1769843, 'cail2018_task1': 1691509, 'dmsc': 1490497, 'baike2018': 1355169, 'alitianchi_renewal': 1256799, 'lang8': 1193457, 'sougo_cs2012_renewal': 1162719, 'baike': 1082976, 'koto_market_classify': 1051973, 'sogouca_summary': 1043520, 'reverse_LCSTS': 991817, 'LCSTS': 991817, 'wenben_zhineng_correct': 990999, 'sogouca_article_generation': 958511, 'PDCFT': 783469, 'Ifeng': 759998, 'paraphrase_pytorch_gpu': 743071, 'couplet': 724915, 'sohu_article_generation': 716584, 'sohu_summary': 716584, 'nlpcc_2018_slu': 647142, 'zhihuwentibiaozhu': 532534, 'CoNaLa_code_generation': 528184, 'THUNews': 518109, 'paraphrase_unorder': 459832, 'phoenix_paraphrasing': 450000, 'CHID': 399886, 'chinese_poetry_collection': 374961, 'CMNLI': 373783, 'chinese_poetry_collection1': 372352, 'cls_2022_renewal': 371521, 'cls_article_generation': 366042, 'cls_summary': 366042, 'csl_clean_keywords_extraction': 356442, 'CSL': 356442, '20221_gxcq': 346008, 'financezhidao': 302913, 'DuIEV2': 294858, 'CRMC2017': 274000, 'people_dairy_2014': 257641, 'wang': 251281, 'RECO_reading_comprehension': 250000, 'hybridset': 244329, 'Dureader': 233480, 'LCQMC': 228745, 'tulinglianbang_classify': 199906, 'ctc': 197634, 'IM_QA': 188775, 'CrimeKgAssitant_master_classify': 183459, 'CrimeKgAssitant_master_qa': 183455, 'ATEC_CCKS': 182477, 'QBCTC': 180000, 'tencent_news_title_classify': 164119, 'tencent_news_content_classify': 161810, 'tencent_news_article_generation': 161810, 'tencent_news_summary': 161810, 'oppo_query_title_similar': 159173, 'OAGD_QA': 157147, 'lajiduanxin_classify': 148751, 'LCTS': 148317, 'CLTS_renewal': 147808, 'tencent_news_keywords_extraction': 141650, 'cic': 135684, 'Chinese_paraphrase_from_Quora_sogou': 133069, 'Chinese_paraphrase_from_Quora_Youdao': 130660, 'AI_judge_classify': 130331, 'teddy6c': 114078, 'zhongxin_news_classify': 112835, 'weibo_senti_100k': 109988, 
#'caipanwenshu_classify': 107237, 'synonyms_paraphrase': 104679, 'AdvertiseGen': 104599, 'liantongzhidao': 101467, 'surgical_qa': 99395, 'bq_corpus': 95478, 'Chinese_idiom_paraphrase': 92063, 'Chinese_PPDB': 90000, 'imcs21_task1': 88606, 'pediatric_qa': 87047, 'zh_book_classify': 85761, 'andriatria_qa': 80678, 'anhuidianxinzhidao': 78067, 'CMedQA': 76977, 'CINLID': 76124, 'Sogou_QA': 75963, 'auto_master_qa': 74486, 'Adversarial_Attack_similar': 73080, 'DIGIX2021': 69454, 'IMCS_V2_DAC_classify': 68327, 'oncology_qa': 64730, 'Chinese_abstractive_article_generation': 63668, 'zhxinli_qa': 63516, 't_news': 57360, 'online_shopping': 56736, 'baiduzhidao_qa': 54114, '2019news_summary_classify': 51141, 'msra_ner-iob2': 49760, 'OCNLI': 47937, 'nlpcc2018task3': 47499, 'PAWS_X': 47001, 'ai_studio_renewal': 45742, 'xinwenbiaotizhaiyao_article_generation': 45630, 'xinwenbiaotizhaiyao_summary': 45630, 'chengyutiankong': 45000, 'iflytek_chinese_match': 45000, 'clothesAdverGen': 45000, 'Chinese_abstractive_summary': 43112, 'zh_bingju_classify': 40727, 'NL2SQL': 39422, 'COTE_mafengwo': 39253, 'CAIL2019_TASK1': 38099, 'caixin_title_news_classify': 37987, 'tang_poetry_renewal': 37963, 'zhihuwenzhang_article_generation': 36000, 'zhihuwenzhang_summary': 36000, 'ASAP_SENT': 35008, '10jqka_news_classify': 34073, 'caixin_summary_news_classify': 33981, 'afqmc': 32634, 'cail2019_track2_order2': 31961, 'yiyuzheng_qa': 31224, 'CNSE': 30304, 'cail2019_track2_order1': 30293, 'NLPCC2014_LSHT': 29867, 'dingxiangyuan_classify': 29266, 'jdjk_query_classify': 28373, 'dingxiangyuan_qa': 28075, 'pengpai_news_title_classify': 28000, 'pengpai_news_content_classify': 27769, 'CNSS': 27663, 'pengpai_news_article_generation': 27475, 'pengpai_news_summary': 27475, 'ccks2021_ydlj': 27285, 'dlner': 26007, 'people_dairy_1998': 25036, 'zh_medicine_qa': 24518, 'jingdong_health_clinical': 24257, 'COTE_dianping': 24058, 'KUAKE_QTR': 22962, 'mrc1819': 22664, 'sinanews_title_classify': 22395, 'IMCS_V2_SR_classify': 22092, 'nlpcc2018_qa': 22037, 'netease_news_classify': 21600, 'DuSQL': 21421, 'CSSCI_summary': 20905, 'non_standard_desease_clinical': 20868, 'CSSCI_keywords_extraction': 20822, 'CCPM_master': 20778, 'CCPM_master_reverse': 20778, 'CSSCI_article_generation': 20448, 'sinanews_content_classify': 19871, 'patent_article_generation': 19839, 'patent_title_abstract': 19839, 'nlpcc2018_task4': 19216, 'CHIP_2019_T2': 19000, 'JEC_QA': 18957, 'cged2016': 18295, 'QQGend': 18103, 'SMP_ECISA': 18013, 'NLPEC_qa': 17773, 'ustinian_article_generation': 17738, 'ustinian_summary': 17738, 'Adversarial_Attack_paraphrase': 17228, '2019news_summary_example_classify': 16879, 'rmrb2020_article_generation': 16865, 'rmrb2020_summary': 16865, 'zhongyiwenxianyuedulijie': 16678, 'zhongyiwenxianwentishengcheng': 16678, 'sohu_news_classify': 16601, 'nonghangzhidao': 15929, 'lawzhidao': 15737, 'CHEF_news_classify': 15652, 'WebQA': 15481, 'ChineseBiomedicalQA': 15336, 'CHIP_STS': 15198, 'NLPCC_Weibo_multi': 15079, 'CCL2019_zh_humor_task2_classify': 15010, 'cail2019_track2_divorce': 14889, 'CCL2019_zh_humor_task1_classify': 14780, 'sohu_news_article_generation': 14600, 'sohu_news_summary': 14600, 'KUAKE_QQR': 14250, 'ccfbdci': 14150, 'C3': 14069, 'haihua2021ydlj': 13881, 'pengpai_news_ot': 13846, 'DuReaderQG_rc': 13820, 'DuReader_robus': 13819, 'pengpai_news_keywords_extraction': 13629, 'TNEWS2106_classify': 12810, 'non_standard_disease_classify2': 12722, 'unknown_source': 12711, '163_cnews_2011_keywords_extraction': 12326, '163_news_article_generation': 11937, 'software21_news_title_classify': 11393, 'software21_news_article_generation': 11297, 'software21_news_summary': 11297, 'IFLYTEK': 11287, 'software21_news_content_classify': 11109, 'waimai_10k': 10987, 'CMeEE': 10616, 'LegalQA': 10329, 'CCF2020-BDCI-NER': 10210, 'cluener_public': 10210, 'CMRC2018': 9642, 'sohu_sts_B_ll': 9517, 'sohu_sts_A_ll': 9516, 'sohu_sts_A_sl': 9481, 'cged2017': 9448, 'sohu_sts_B_sl': 9444, 'sohu_sts_A_ss': 9376, 'sohu_sts_B_ss': 9368, 'nlpcc2016_dbqa': 9198, 'nlpcc2017_qa': 9198, 'nlpcc2014_task2': 9052, 'food_safe_classify': 9000, 'NLPCC14-SC': 9000, 'nlpcc2020-AutoIE': 9000, 'bank': 9000, 'jiujiujiankang_qa': 9000, '163_news_summary': 8755, 'COVID_TEXT': 8747, 'SENTI_RATIONALE': 8696, 'cail2019_track2_loan': 8244, 'cail2019_track2_labor': 8094, 'COTE_baidu': 8083, 'CHEF_claim_classify': 8002, 'zh_humor_degree_classify': 7246, 'ecommerce': 7198, 'zhongqing_news_classify': 7197, 'DRCD2018': 7155, 'ChnsentiCorp': 7033, 'CCL_GCRC': 6644, 'KUAKE_QIC': 6581, 'zh_humor_category_classify': 6578, 'CSpider': 6491, 'sighan': 6176, 'tencent_news_ot': 5943, 'ccks_kbqa': 5706, 'space2021_task2': 5389, 'cail2022_sfyz_summary': 5235, 'CSTS_0-5_Chinese-STS-B': 4981, 'CSTS_Chinese-STS-B': 4981, 'CAIL2020_TASK1': 4550, 'ccks2022_ydlj': 4500, 'yiqingzhengwuyuedulijie': 4500, 'similar5000': 4500, 'finance_negative_classify': 4491, 'public_query_classify': 4461, 'resume': 4284, 'Internet_News': 4104, 'clinical_terminology_standartization': 4000, 'clinical_terminology_standartization_reverse': 4000, 'space2021_task1': 3837, 'Chase': 3749, 'xianghewang_qa': 3724, 'CCFBDCI2020': 3630, 'auto_forum_classify': 3564, 'zhihuishuqa_classify_with_answer': 3368, 'zhihuishuqa_classify': 3368, 'wanfang_article_generation': 3343, 'wanfang_title_generation': 3343, 'green_news_classify': 3318, 'OCR': 3275, 'CCL2018_Chinese_Metaphor_task2': 3270, 'baoxianzhidao': 3241, 'software20_label_content_classify2': 3155, 'software20_label_title_classify2': 3155, 'mmc': 3148, 'software20_label_title_classify1': 3130, 'software20_label_content_classify1': 3130, 'software20_label_article_generation': 3130, 'software20_label_summary': 3130, 'yizhiwenhuayuedulijie': 2949, 'yizhiwenhuawentishengcheng': 2949, 'shijing': 2707, 'wanfangSpider': 2645, 'netease_news2_classify': 2515, 'reverse_CRMC2018': 2283, 'ChineseSQuAD': 2109, 'SMP2018_ECDT_classify': 2099, 'cged2015': 2095, 'cged2021': 2081, 'ccks2017_task2': 2006, 'ccks2017_task21': 2006, 'zhongwenweibolichang': 1997, 'sishu': 1960, 'med_intent_classify': 1924, 'reverse_DRCD': 1860, 'data_article_generation': 1800, 'data_summary': 1800, 'BosonNLP_NER_6C': 1799, 'Lyra_code_generation_zh': 1600, 'Lyra_code_generation_en': 1600, 'mengxue': 1435, 'cged2014': 1431, 'finance_sina': 1421, 'weibo_ner': 1350, 'weiboNER_2nd_conll': 1350, 'DuReader_checklist': 1334, 'ccks2019_task1': 1241, 'wanchuang': 1129, 'paddle_similar': 909, 'cged2020': 775, 'ccks2018_task1': 717, 'wsc': 467, 'gov_office_text_correct': 450, 'cged2018': 362}
##task_weights_321 = {'caipanwenshu_classify': 107237, 'synonyms_paraphrase': 104679, 'AdvertiseGen': 104599}
#zero_eval_task = set(['software20_label_title_classify2', 'software20_label_title_classify1', 'software20_label_content_classify1', 
#	'netease_news2_classify', 'SMP2018_ECDT_classify', 'zhongwenweibolichang', 'med_intent_classify', 'yiqingzhengwuyuedulijie', 
#	'yizhiwenhuayuedulijie', 'ChineseSQuAD', 'sohu_news_article_generation', 'software21_news_article_generation', 
#	'software20_label_article_generation', 'NLPCC14-SC', 'SENTI_RATIONALE', 'unknown_source', 'Adversarial_Attack_similar', 
#	'jiujiujiankang_qa', 'ccks_kbqa', 'xianghewang_qa', 'software21_news_summary', 'software20_label_summary', 
#	'163_cnews_2011_keywords_extraction', 'tencent_news_ot', 'hybridset', 'clinical_terminology_standartization_reverse', 
#	'clinical_terminology_standartization', 'yizhiwenhuawentishengcheng', 'sishu', 'mengxue', 'cail2019_track2_labor'])
#
#task_weights = {key: value for key, value in task_weights_321.items() if key not in zero_eval_task}
#task_weights_169 = {'kg_clue': 19731218, 'LCCD': 14677547, 'nlpcc2015_task5': 8548913, 'translation2019zhen': 5131434, 'translation2019enzh': 5131434, 'wmt_20_zh_en': 5000002, 'wmt_20_en_zh': 5000002, 'Medical_Dialogue_System': 4503385, 'wudao_article_generation': 4255462, 'web_text': 4111080, 'PyTorrent_code_generation_docstring': 2258188, 'pchatbotL_01': 2140668, 'product_desc_ali': 2109187, 'cnki_article_generation': 1966982, 'cnki_summary': 1966982, 'alitianchi_renewal': 1256799, 'lang8': 1193457, 'sougo_cs2012_renewal': 1162719, 'movie_chats': 1081246, 'sogouca_summary': 1043520, 'pchatbotW': 1011784, 'reverse_LCSTS': 991817, 'LCSTS': 991817, 'wenben_zhineng_correct': 990999, 'sogouca_article_generation': 958511, 'Chinese_medical_dialogue': 772098, 'paraphrase_pytorch_gpu': 743071, 'couplet': 724915, 'sohu_article_generation': 716584, 'sohu_summary': 716584, 'pchatbotL': 696485, 'nlpcc_2018_slu': 647142, 'CoNaLa_code_generation': 528184, 'QG_data': 476108, 'paraphrase_unorder': 459832, 'phoenix_paraphrasing': 450000, 'zhongxueshengzuowen_conv': 447320, 'chinese_poetry_collection': 374961, 'chinese_poetry_collection1': 372352, 'cls_2022_renewal': 371521, 'cls_article_generation': 366042, 'cls_summary': 366042, 'financezhidao': 302913, 'wang': 251281, 'hybridset': 244329, 'ctc': 197634, 'IM_QA': 188775, 'Dulemon_self': 187997, 'CrimeKgAssitant_master_qa': 183455, 'shicidaquan_conv': 183078, 'Chinese_personal_chat': 178007, 'MedDG': 169280, 'tencent_news_article_generation': 161810, 'tencent_news_summary': 161810, 'OAGD_QA': 157147, 'LCTS': 148317, 'CLTS_renewal': 147808, 'Chinese_paraphrase_from_Quora_sogou': 133069, 'Chinese_paraphrase_from_Quora_Youdao': 130660, 'teddy6c': 114078, 'synonyms_paraphrase': 104679, 'AdvertiseGen': 104599, 'liantongzhidao': 101467, 'surgical_qa': 99395, 'Chinese_idiom_paraphrase': 92063, 'Chinese_PPDB': 90000, 'Du_Conv': 89899, 'pediatric_qa': 87047, 'chinese_sentence_function': 85897, 'andriatria_qa': 80678, 'anhuidianxinzhidao': 78067, 'CMedQA': 76977, 'auto_master_qa': 74486, 'risawoz': 67290, 'oncology_qa': 64730, 'Chinese_abstractive_article_generation': 63668, 'zhxinli_qa': 63516, 'baiduzhidao_qa': 54114, 'Du_Recial': 49938, 'nlpcc2018task3': 47499, 'SSD_NAME': 46701, 'SSD_PHONE': 46172, 'ai_studio_renewal': 45742, 'xinwenbiaotizhaiyao_article_generation': 45630, 'xinwenbiaotizhaiyao_summary': 45630, 'clothesAdverGen': 45000, 'SSD_ID': 44433, 'Chinese_abstractive_summary': 43112, 'crosswoz': 42346, 'luge_Diamante': 39433, 'NL2SQL': 39422, 'tang_poetry_renewal': 37963, 'zhihuwenzhang_article_generation': 36000, 'zhihuwenzhang_summary': 36000, 'yiyuzheng_qa': 31224, 'Bitod_main': 28873, 'dingxiangyuan_qa': 28075, 'pengpai_news_article_generation': 27475, 'pengpai_news_summary': 27475, 'SSD_PLATE': 27061, 'zh_medicine_qa': 24518, 'zhdd': 24514, 'nlpcc2018_qa': 22037, 'DuSQL': 21421, 'CSSCI_summary': 20905, 'CCPM_master': 20778, 'CCPM_master_reverse': 20778, 'CSSCI_article_generation': 20448, 'patent_article_generation': 19839, 'patent_title_abstract': 19839, 'Dulemon_both': 19437, 'JEC_QA': 18957, 'cged2016': 18295, 'NLPEC_qa': 17773, 'ustinian_article_generation': 17738, 'ustinian_summary': 17738, 'Adversarial_Attack_paraphrase': 17228, 'rmrb2020_article_generation': 16865, 'rmrb2020_summary': 16865, 'nonghangzhidao': 15929, 'lawzhidao': 15737, 'ChineseBiomedicalQA': 15336, 'sohu_news_article_generation': 14600, 'sohu_news_summary': 14600, 'kd_conv_film': 14366, 'CPED': 14249, '163_news_article_generation': 11937, 'software21_news_article_generation': 11297, 'software21_news_summary': 11297, 'LegalQA': 10329, 'kd_conv_music': 9592, 'cged2017': 9448, 'Dusinc': 9413, 'kd_conv_travel': 9280, 'nlpcc2016_dbqa': 9198, 'nlpcc2017_qa': 9198, 'jiujiujiankang_qa': 9000, '163_news_summary': 8755, 'CSpider': 6491, 'sighan': 6176, 'ccks_kbqa': 5706, 'cail2022_sfyz_summary': 5235, 'clinical_terminology_standartization': 4000, 'clinical_terminology_standartization_reverse': 4000, 'covid-dialogue': 3922, 'm3ed': 3908, 'Chase': 3749, 'xianghewang_qa': 3724, 'wanfang_article_generation': 3343, 'wanfang_title_generation': 3343, 'OCR': 3275, 'baoxianzhidao': 3241, 'software20_label_article_generation': 3130, 'software20_label_summary': 3130, 'shijing': 2707, 'cged2015': 2095, 'cged2021': 2081, 'sishu': 1960, 'data_article_generation': 1800, 'data_summary': 1800, 'Lyra_code_generation_en': 1600, 'Lyra_code_generation_zh': 1600, 'mengxue': 1435, 'cged2014': 1431, 'cged2020': 775, 'gov_office_text_correct': 450, 'cged2018': 362, 'person_role': 101, 'qingjiatiao': 50}
task_weights_170 = {'kg_clue': 19731218, 'LCCD': 14677547, 'nlpcc2015_task5': 8548913, 'translation2019zhen': 5131434, 'translation2019enzh': 5131434, 'wmt_20_zh_en': 5000002, 'wmt_20_en_zh': 5000002, 'Medical_Dialogue_System': 4503385, 'wudao_article_generation': 4255462, 'web_text': 4111080, 'PyTorrent_code_generation_docstring': 2258188, 'pchatbotL_01': 2140668, 'ganrao_shuju': 2109744, 'product_desc_ali': 2109187, 'cnki_article_generation': 1966982, 'cnki_summary': 1966982, 'alitianchi_renewal': 1256799, 'lang8': 1193457, 'sougo_cs2012_renewal': 1162719, 'movie_chats': 1081246, 'sogouca_summary': 1043520, 'pchatbotW': 1011784, 'reverse_LCSTS': 991817, 'LCSTS': 991817, 'wenben_zhineng_correct': 990999, 'sogouca_article_generation': 958511, 'Chinese_medical_dialogue': 772098, 'paraphrase_pytorch_gpu': 743071, 'couplet': 724915, 'sohu_article_generation': 716584, 'sohu_summary': 716584, 'pchatbotL': 696485, 'nlpcc_2018_slu': 647142, 'CoNaLa_code_generation': 528184, 'QG_data': 476108, 'paraphrase_unorder': 459832, 'phoenix_paraphrasing': 450000, 'zhongxueshengzuowen_conv': 447320, 'chinese_poetry_collection': 374961, 'chinese_poetry_collection1': 372352, 'cls_2022_renewal': 371521, 'cls_article_generation': 366042, 'cls_summary': 366042, 'financezhidao': 302913, 'wang': 251281, 'hybridset': 244329, 'ctc': 197634, 'IM_QA': 188775, 'Dulemon_self': 187997, 'CrimeKgAssitant_master_qa': 183455, 'shicidaquan_conv': 183078, 'Chinese_personal_chat': 178007, 'MedDG': 169280, 'tencent_news_article_generation': 161810, 'tencent_news_summary': 161810, 'OAGD_QA': 157147, 'LCTS': 148317, 'CLTS_renewal': 147808, 'Chinese_paraphrase_from_Quora_sogou': 133069, 'Chinese_paraphrase_from_Quora_Youdao': 130660, 'teddy6c': 114078, 'synonyms_paraphrase': 104679, 'AdvertiseGen': 104599, 'liantongzhidao': 101467, 'surgical_qa': 99395, 'Chinese_idiom_paraphrase': 92063, 'Chinese_PPDB': 90000, 'Du_Conv': 89899, 'pediatric_qa': 87047, 'chinese_sentence_function': 85897, 'andriatria_qa': 80678, 'anhuidianxinzhidao': 78067, 'CMedQA': 76977, 'auto_master_qa': 74486, 'risawoz': 67290, 'oncology_qa': 64730, 'Chinese_abstractive_article_generation': 63668, 'zhxinli_qa': 63516, 'baiduzhidao_qa': 54114, 'Du_Recial': 49938, 'nlpcc2018task3': 47499, 'SSD_NAME': 46701, 'SSD_PHONE': 46172, 'ai_studio_renewal': 45742, 'xinwenbiaotizhaiyao_article_generation': 45630, 'xinwenbiaotizhaiyao_summary': 45630, 'clothesAdverGen': 45000, 'SSD_ID': 44433, 'Chinese_abstractive_summary': 43112, 'crosswoz': 42346, 'luge_Diamante': 39433, 'NL2SQL': 39422, 'tang_poetry_renewal': 37963, 'zhihuwenzhang_article_generation': 36000, 'zhihuwenzhang_summary': 36000, 'yiyuzheng_qa': 31224, 'Bitod_main': 28873, 'dingxiangyuan_qa': 28075, 'pengpai_news_article_generation': 27475, 'pengpai_news_summary': 27475, 'SSD_PLATE': 27061, 'zh_medicine_qa': 24518, 'zhdd': 24514, 'nlpcc2018_qa': 22037, 'DuSQL': 21421, 'CSSCI_summary': 20905, 'CCPM_master': 20778, 'CCPM_master_reverse': 20778, 'CSSCI_article_generation': 20448, 'patent_article_generation': 19839, 'patent_title_abstract': 19839, 'Dulemon_both': 19437, 'JEC_QA': 18957, 'cged2016': 18295, 'NLPEC_qa': 17773, 'ustinian_article_generation': 17738, 'ustinian_summary': 17738, 'Adversarial_Attack_paraphrase': 17228, 'rmrb2020_article_generation': 16865, 'rmrb2020_summary': 16865, 'nonghangzhidao': 15929, 'lawzhidao': 15737, 'ChineseBiomedicalQA': 15336, 'sohu_news_article_generation': 14600, 'sohu_news_summary': 14600, 'kd_conv_film': 14366, 'CPED': 14249, '163_news_article_generation': 11937, 'software21_news_article_generation': 11297, 'software21_news_summary': 11297, 'LegalQA': 10329, 'kd_conv_music': 9592, 'cged2017': 9448, 'Dusinc': 9413, 'kd_conv_travel': 9280, 'nlpcc2016_dbqa': 9198, 'nlpcc2017_qa': 9198, 'jiujiujiankang_qa': 9000, '163_news_summary': 8755, 'CSpider': 6491, 'sighan': 6176, 'ccks_kbqa': 5706, 'cail2022_sfyz_summary': 5235, 'clinical_terminology_standartization': 4000, 'clinical_terminology_standartization_reverse': 4000, 'covid-dialogue': 3922, 'm3ed': 3908, 'Chase': 3749, 'xianghewang_qa': 3724, 'wanfang_article_generation': 3343, 'wanfang_title_generation': 3343, 'OCR': 3275, 'baoxianzhidao': 3241, 'software20_label_article_generation': 3130, 'software20_label_summary': 3130, 'shijing': 2707, 'cged2015': 2095, 'cged2021': 2081, 'sishu': 1960, 'data_article_generation': 1800, 'data_summary': 1800, 'Lyra_code_generation_en': 1600, 'Lyra_code_generation_zh': 1600, 'mengxue': 1435, 'cged2014': 1431, 'cged2020': 775, 'gov_office_text_correct': 450, 'cged2018': 362, 'person_role': 101, 'qingjiatiao': 50}

task_weights_170["person_role"] = 1000
task_weights = task_weights_170

limit_num = 10000000
task_glob_num_str = "[0][0][0][0-9][0-9][0-9]"
#task_glob_num_str = "*"
logging.info(f"limit_num:{limit_num}, train task size:{len(task_weights)}")
#logging.info(f"zero_eval_task:{len(zero_eval_task)}, all_task size:{len(task_weights_321)}, train task size:{len(task_weights)}")
logging.info(f"all_task size:{len(task_weights)}, train task size:{len(task_weights)}")
# 各个数据集注册任务以及分配权重
i = 0
for item in task_weights.keys():
	i += 1
	if i <= 1000:
		seqio.TaskRegistry.add(item.replace("-", "_"),
			# 定义数据源(传入了一个函数，这个函数的返回就是数据源)
			source=seqio.FunctionDataSource(
				dataset_fn=functools.partial(customize_finetuning_dataset_fn, dataset_name=item,
											glob_num_str=task_glob_num_str),
				splits=["train", "dev"]
			),
			# 定义数据预处理器（数据送进model之前需要做的处理）
			preprocessors=[
				customize_finetuning_preprocessor,
				seqio.preprocessors.tokenize_and_append_eos,
			],
			# 定义数据后处理器 (数据从model输出之后需要做的处理）
			#postprocess_fn=t5.data.postprocessors.lower_text,
			# 定义评价指标
			metric_fns=[metrics.accuracy], # [customize_metric],
			# 输出token解码方式
			output_features=DEFAULT_OUTPUT_FEATURES,
		)
	else:
		seqio.TaskRegistry.add(item.replace("-", "_"),
			# 定义数据源(传入了一个函数，这个函数的返回就是数据源)
			source=seqio.FunctionDataSource(
				dataset_fn=functools.partial(customize_finetuning_dataset_fn, dataset_name=item,
											glob_num_str=task_glob_num_str),
				splits=["train"]
			),
			# 定义数据预处理器（数据送进model之前需要做的处理）
			preprocessors=[
				customize_finetuning_preprocessor,
				seqio.preprocessors.tokenize_and_append_eos,
			],
			# 定义数据后处理器 (数据从model输出之后需要做的处理）
			#postprocess_fn=t5.data.postprocessors.lower_text,
			# 定义评价指标
			metric_fns=[metrics.accuracy], # [customize_metric],
			# 输出token解码方式
			output_features=DEFAULT_OUTPUT_FEATURES,
		)


	'''
	seqio.TaskRegistry.add(item.replace("-", "_") + "_decoder",
		# 定义数据源(传入了一个函数，这个函数的返回就是数据源)
		source=seqio.FunctionDataSource(
			dataset_fn=functools.partial(customize_finetuning_dataset_fn, dataset_name=item,
										glob_num_str=task_glob_num_str),
			splits=["train"]
		),
		# 定义数据预处理器（数据送进model之前需要做的处理）
		preprocessors=[
			customize_finetuning_preprocessor_decoder,
			seqio.preprocessors.tokenize_and_append_eos,
		],
		# 定义数据后处理器 (数据从model输出之后需要做的处理）
		#postprocess_fn=t5.data.postprocessors.lower_text,
		# 定义评价指标
		metric_fns=[metrics.accuracy], # [customize_metric],
		# 输出token解码方式
		output_features=DEFAULT_OUTPUT_FEATURES,
	)

	seqio.TaskRegistry.add(item.replace("-", "_") + "_encoder",
		# 定义数据源(传入了一个函数，这个函数的返回就是数据源)
		source=seqio.FunctionDataSource(
			dataset_fn=functools.partial(customize_finetuning_dataset_fn, dataset_name=item,
										glob_num_str=task_glob_num_str),
			splits=["train"]
		),
		preprocessors=[
			customize_pretrain_preprocessor_encoder,
			functools.partial(
				t5.data.preprocessors.rekey, key_map={
					"inputs": None,
					"targets": "targets", # "text"
				}),
			seqio.preprocessors.tokenize,
			seqio.CacheDatasetPlaceholder(),
			t5.data.preprocessors.span_corruption, # TODO can try this later: t5.data.preprocessors.lm
			#t5.data.preprocessors.prefix_lm,
			seqio.preprocessors.append_eos_after_trim,
    	],
    output_features=DEFAULT_OUTPUT_FEATURES,
	# output_features=t5.data.Feature(vocabulary=t5.data.SentencePieceVocabulary(my_custom_vocab_model_path)),

	metric_fns=[])
	'''

#dataset_name = "iflytek_chinese_match"
dataset_name = "*"
pretrain_task_glob_num_str = "[0][0][0][0-9][0-9][0-9]"
seqio.TaskRegistry.add( "pretrain_mt_decoder",
	# 定义数据源(传入了一个函数，这个函数的返回就是数据源)
	source=seqio.FunctionDataSource(
		dataset_fn=functools.partial(customize_finetuning_dataset_fn, dataset_name=dataset_name,
									glob_num_str=pretrain_task_glob_num_str),
		splits=["train", "dev"]
	),
	# 定义数据预处理器（数据送进model之前需要做的处理）
	preprocessors=[
		customize_finetuning_preprocessor_decoder,
		seqio.preprocessors.tokenize_and_append_eos,
	],
	# 定义数据后处理器 (数据从model输出之后需要做的处理）
	#postprocess_fn=t5.data.postprocessors.lower_text,
	# 定义评价指标
	#metric_fns=[metrics.accuracy], # [customize_metric],
	metric_fns=[], # [customize_metric],
	# 输出token解码方式
	output_features=DEFAULT_OUTPUT_FEATURES,
)

seqio.TaskRegistry.add( "pretrain_mt_encoder",
	# 定义数据源(传入了一个函数，这个函数的返回就是数据源)
	source=seqio.FunctionDataSource(
		dataset_fn=functools.partial(customize_finetuning_dataset_fn, dataset_name=dataset_name,
									glob_num_str=pretrain_task_glob_num_str),
		splits=["train", "dev"]
	),
	preprocessors=[
		customize_pretrain_preprocessor_encoder,
		functools.partial(
			t5.data.preprocessors.rekey, key_map={
				"inputs": None,
				"targets": "targets", # "text"
			}),
		seqio.preprocessors.tokenize,
		seqio.CacheDatasetPlaceholder(),
		t5.data.preprocessors.span_corruption, # TODO can try this later: t5.data.preprocessors.lm
		#t5.data.preprocessors.prefix_lm,
		seqio.preprocessors.append_eos_after_trim,
	],
    output_features=DEFAULT_OUTPUT_FEATURES,
    # output_features=t5.data.Feature(vocabulary=t5.data.SentencePieceVocabulary(my_custom_vocab_model_path)),

    #metric_fns=[metrics.accuracy],
    metric_fns=[],
)

seqio.TaskRegistry.add( "pretrain_mt_encoder_decoder",
	# 定义数据源(传入了一个函数，这个函数的返回就是数据源)
	source=seqio.FunctionDataSource(
		dataset_fn=functools.partial(customize_finetuning_dataset_fn, dataset_name=dataset_name,
									glob_num_str=pretrain_task_glob_num_str),
		splits=["train", "dev"]
	),
	preprocessors=[
		customize_pretrain_preprocessor_encoder_decoder,
		functools.partial(
			t5.data.preprocessors.rekey, key_map={
				"inputs": None,
				"targets": "targets", # "text"
			}),
		seqio.preprocessors.tokenize,
		seqio.CacheDatasetPlaceholder(),
		#t5.data.preprocessors.span_corruption, # TODO can try this later: t5.data.preprocessors.lm
		t5.data.preprocessors.prefix_lm,
		seqio.preprocessors.append_eos_after_trim,
	],
    output_features=DEFAULT_OUTPUT_FEATURES,
    # output_features=t5.data.Feature(vocabulary=t5.data.SentencePieceVocabulary(my_custom_vocab_model_path)),

    metric_fns=[],
)


# 分配权重
import math
def _weight(v):
	v = limit_num if v > limit_num else v
	return int(math.sqrt(v))

clueai_mt_weights = [(k.replace("-", "_"), _weight(v)) for k, v in task_weights.items()]
logging.info(f"task num: {len(clueai_mt_weights)}, clueai_mt_weights:{clueai_mt_weights}")
seqio.MixtureRegistry.add(
	"clueai_mt",
	clueai_mt_weights
)

sum_weight = sum(_weight(v) for k, v in task_weights.items())
logging.info(f"sum weight: {sum_weight}")
#clueai_mt_weights_encoder = [(k.replace("-", "_") + "_encoder", _weight(v)) for k, v in task_weights.items()]
#clueai_mt_weights_decoder = [(k.replace("-", "_") + "_decoder", _weight(v)) for k, v in task_weights.items()]
#clueai_mt_weights_all = clueai_mt_weights + clueai_mt_weights_encoder + clueai_mt_weights_decoder
clueai_mt_weights_all = clueai_mt_weights + [("pretrain_mt_decoder", sum_weight//4), ("pretrain_mt_encoder", sum_weight//4), ("pretrain_mt_encoder_decoder", sum_weight//4)]
seqio.MixtureRegistry.add(
	# Mixture名称
	"clueai_mt_weights_all", # ValueError: No Task or Mixture found with name 'clueai_corpus'. Available:
	# 组成Mixture的Tasks
	clueai_mt_weights_all  # , "task_2"
)

# 预训练的语料任务注册
seqio.TaskRegistry.add(
    "clueai_corpus",
    source=seqio.FunctionDataSource(
		dataset_fn=functools.partial(customize_pretrain_dataset_fn, lang="chinese")	,
		splits=["train"]  # , "dev", "test"]
	),
    preprocessors=[
        customize_pretrain_preprocessor,
        functools.partial(
			t5.data.preprocessors.rekey, key_map={
                "inputs": None,
                "targets": "targets", # "text"
            }),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
		#t5.data.preprocessors.span_corruption, # TODO can try this later: t5.data.preprocessors.lm
		t5.data.preprocessors.prefix_lm,
        seqio.preprocessors.append_eos_after_trim,

    ],
    output_features=DEFAULT_OUTPUT_FEATURES,
	# output_features=t5.data.Feature(vocabulary=t5.data.SentencePieceVocabulary(my_custom_vocab_model_path)),

	metric_fns=[])

# 使用整理好的100个数据集进行预训练
seqio.TaskRegistry.add(
    "clueai_mt_pretain_corpus",
    source=seqio.FunctionDataSource(
		dataset_fn=functools.partial(customize_pretrain_dataset_fn_mt, lang="chinese")	,
		splits=["train"]  # , "dev", "test"]
	),
    preprocessors=[
        customize_pretrain_preprocessor,
        functools.partial(
			t5.data.preprocessors.rekey, key_map={
                "inputs": None,
                "targets": "targets", # "text"
            }),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
		t5.data.preprocessors.span_corruption, # TODO can try this later: t5.data.preprocessors.lm
		#t5.data.preprocessors.prefix_lm,
        seqio.preprocessors.append_eos_after_trim,

    ],
    output_features=DEFAULT_OUTPUT_FEATURES,
	# output_features=t5.data.Feature(vocabulary=t5.data.SentencePieceVocabulary(my_custom_vocab_model_path)),

	metric_fns=[])

# 混合多个任务
#clueai_corpus_mt_weights = [("clueai_corpus", 2*sum_weight), ("clueai_mt_pretain_corpus", sum_weight)] + clueai_mt_weights
clueai_corpus_mt_weights = [("clueai_corpus", sum_weight//2)]  + clueai_mt_weights_all
#clueai_corpus_mt_weights = clueai_mt_weights_all
logging.info(f"clueai_corpus_mt_weights:{clueai_corpus_mt_weights}")
seqio.MixtureRegistry.add(
	# Mixture名称
	"clueai_corpus_mt", # ValueError: No Task or Mixture found with name 'clueai_corpus'. Available:
	# 组成Mixture的Tasks
	clueai_corpus_mt_weights
)