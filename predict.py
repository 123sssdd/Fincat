from curses.ascii import isdigit
import json 
import os
from glm_components.query_company import BM25EnityExtractor
from glm_components import finance_table_query, company_info_query, personal_query, open_query, common_query, complex_query
from argparse import ArgumentParser
from collections import Counter
import pandas as pd
import regex as re
from finetune.table_qa.classifier import finance_compute_feaures, finance_features, load_equations, financial_alias_inv_mapping
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers import BitsAndBytesConfig
from itertools import chain
import torch
from collections import defaultdict
import gc
from pathlib import Path
from transformers.utils import logging
import time
import multiprocessing as mp 
from fastapi import FastAPI
import asyncio
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
model, tokenizer = None, None
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许的前端地址
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有 HTTP 方法
    allow_headers=["*"],  # 允许所有请求头
)
    
manager = mp.Manager()
client_question = manager.Value('s', '')  # 创建可以跨进程共享的变量
client_answer = manager.Value('s', '')
get_question_event = manager.Event()
get_answer_event = manager.Event()

# tokenizer.padding_side = 'left'

def load_model():
    model_path = "model/chatglm2-6b"
    checkpoint_path = "model/classifier/sql_enhance_checkpoint"
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config.prefix_projection = False
    config.pre_seq_len = 128
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, config=config, trust_remote_code=True)
    prefix_state_dict = torch.load(os.path.join(checkpoint_path, "pytorch_model.bin"))
    new_prefix_state_dict = {}
    for k, v in prefix_state_dict.items():
        if k.startswith("transformer.prefix_encoder."):
            new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
    model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)
    model = model.half()
    model = model.cuda()
    return model, tokenizer


# def load_model():
#     model_path = "model/chatglm-6b-int4"
#     tokenizer = AutoTokenizer.from_pretrained(, trust_remote_code=True, revision="")
#     model = AutoModel.from_pretrained("D:\\data\\llm\\chatglm-6b-int4", trust_remote_code=True, revision="").half().cuda()
#     model = model.eval()
#     return model, tokenizer




def get_response_classify(questions):
    global model, tokenizer
    if model is None:
        model, tokenizer = load_model()
    inputs = tokenizer(questions, return_tensors="pt", max_length=512, truncation=True, padding=True)
    inputs = inputs.to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=256, do_sample=True, top_p=0.7, temperature=0.95, num_beams=4)
    # outputs = model.generate(**inputs, max_new_tokens=256, num_beams=4, early_stopping=True)
    response = tokenizer.batch_decode(outputs)
    response = [res[res.index("答：")+2:].strip() for res in response]
    final_response = []
    for res in response:
        try:
            data_json = json.loads(res)
            final_response.append(data_json)
        except json.JSONDecodeError:
            print(res)
            final_response.append(None)
    print("get_response_classify")
    print(final_response)
    return final_response

def format_conversation(question):
    return f"[Round 1]\n\n问：提取以下句子的问题类型年份、公司名称，如果是开放问题，提取出对应的财报章节，如果是查询问题，请提供SQL查询和回答模板，如果是财务类问题，提取出对应的财务指标，对非复杂计算的指标，请给出回答模板，以json形式回答:{question}\n\n答："

def dump_classification_results(file_path=None, refresh=True,question=None):
    if file_path is None:
        file_path = "finetune/table_qa/data/classification.jsonl"
    if os.path.exists(file_path) and not refresh:
        return
    global model, tokenizer
    if model is None:
        model, tokenizer = load_model()
        model.eval()
    from tqdm import tqdm

    answers = []
    with torch.no_grad():
        question_list = []
        question_list.append(question)
        pbar = enumerate(tqdm(question_list))
        batch_data = []
        for idx, question in pbar:
            batch_data.append(format_conversation(question))
            success = False
            if len(batch_data) == 16 or idx == len(question_list) - 1:
                responses = get_response_classify(batch_data)
                answers.extend(responses)
                batch_data = []
    max_try = 0
    failed_cnt = 0
    while len([ans for ans in answers if ans is None]) != 0:
        failed_index = answers.index(None)
        print(f"failed:{failed_index}")
        new_response = get_response_classify([format_conversation(question_list[failed_index])])[0]
        tries = 0
        while new_response is None and tries < max_try:
            new_response = get_response_classify([format_conversation(question_list[failed_index])])[0]
            tries += 1
        answers[failed_index] = new_response
        if tries == max_try:
            answers[failed_index] = {"类型": "失败"}
            failed_cnt += 1
            print(question_list[failed_index])
    with open(file_path, 'w', encoding="utf8") as fp:
        for ans in answers:
            if ans is not None and isinstance(ans, dict) and '关键词' in ans and len(ans['关键词']) == 1 and '类型' in ans:
                if "外文名称" in ans["关键词"][0] and ans['类型'] != '公司问题':
                    ans['类型'] = '公司问题'
                if isinstance(ans["关键词"][0], str) and ans["关键词"][0] in ['现金流', '现金流量'] and ans['类型'] != '开放问题':
                    ans['类型'] = '开放问题'
                    ans['关键词'][0] = "现金流"
            fp.write(json.dumps(ans, ensure_ascii=False) + "\n")
    # model.cpu()
    # del model
    # model = None
    return answers
    

logger = logging.get_logger(__name__)


standard_financial_terms = list(set(finance_features + list(chain(*[key_word_list for key_word_list in load_equations().values()]))))


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--enable_past_year", action="store_true", default=True)
    parser.add_argument('--refresh_classification', action='store_true')
    parser.add_argument('--batch_size', default=1, type=int)
    return parser.parse_args()


args = get_args()


def load_questions():
    """_summary_
    加载所有的问题
    """
    quests = []
    with open(os.path.join(os.path.dirname(__file__), "data/C-list-question.json"), 'r', encoding='utf8') as fp:
        for line in fp.readlines():
            quests.append(json.loads(line)['question'])
    return quests


def load_extracted():
    result = []
    for line in open('finetune/table_qa/data/classification.jsonl', 'r', encoding='utf8').readlines():
        try:
            json_data = json.loads(line)
        except:
            json_data = []
        result.append(json_data)
    for idx, res in enumerate(result):
        if isinstance(res, list):
            result[idx] = {
                "类型": "未知"
            }
    print(result[0])
    return result


def load_excels():
    header_set = defaultdict(lambda :0)
    def verify_header(header):
        return tuple(header) in {
            ('项目', '2021年12月31日', '2020年12月31日'), 
            ('项目', '2020年12月31日', '2019年12月31日'), 
            ('项目', '2019年12月31日', '2018年12月31日'),
            ('项目', '2019年12月31日'),
            ('项目', '2020年12月31日'),
            ('项目', '2021年12月31日'),
            ('项目', '2021年度', '2020年度'),
            ('项目', '2020年度', '2019年度'),
            ('项目', '2019年度', '2018年度'),
        }

    # attribute_fix_pattern = r"^([（\(].*[\)）])|([一二三四五六七八九十]*、)|(其中[\s:：]*)|([加减][\s:：]*)|[0-9]*|([^\u4e00-\u9fa5]*)"
    attribute_fix_pattern = r"([（\(].*[\)）])"
    attribute_list = []
    def preprocess_excels(excel, year):
        year_0, year_1 = str(int(year) - 1), year
        finance_dict = defaultdict(lambda :{})
        if '合并资产负债表' in excel and len(excel['合并资产负债表']) > 0 and verify_header(excel['合并资产负债表'][0]):
            # year_0, year_1 = excel['合并资产负债表'][0][2][:4], excel['合并资产负债表'][0][1][:4]
            for line in excel['合并资产负债表'][1:]:
                line[0] = re.sub(attribute_fix_pattern, "", line[0].replace(" ","").strip())
                # line[0] = line[0].replace(" ", "").strip()
                if line[0] in financial_alias_inv_mapping:
                    line[0] = financial_alias_inv_mapping[line[0]]
                attribute_list.append(line[0])
                if len(line) == 2:
                    if line[0] not in finance_dict[year] or finance_dict[year][line[0]] == "":
                        finance_dict[year][line[0]] = line[-1]
                else:
                    if line[0] not in finance_dict[year_0] or finance_dict[year_0][line[0]] == "":
                        finance_dict[year_0][line[0]] = line[-1]
                    if line[0] not in finance_dict[year_1] or finance_dict[year_1][line[0]] == "":
                        finance_dict[year_1][line[0]] = line[-2]
        if '合并现金流量表' in excel and len(excel['合并现金流量表']) > 0 and verify_header(excel['合并现金流量表'][0]):
            # year_0, year_1 = excel['合并现金流量表'][0][2][:4], excel['合并现金流量表'][0][1][:4]
            for line in excel['合并现金流量表'][1:]:
                line[0] = re.sub(attribute_fix_pattern, "", line[0].replace(" ","").strip())
                # line[0] = line[0].replace(" ", "").strip()
                if line[0] in financial_alias_inv_mapping:
                    line[0] = financial_alias_inv_mapping[line[0]]
                attribute_list.append(line[0])
                if len(line) == 2:
                    if line[0] not in finance_dict[year] or finance_dict[year][line[0]] == "":
                        finance_dict[year][line[0]] = line[-1]
                else:
                    if line[0] not in finance_dict[year_0] or finance_dict[year_0][line[0]] == "":
                        finance_dict[year_0][line[0]] = line[-1]
                    if line[0] not in finance_dict[year_1] or finance_dict[year_1][line[0]] == "":
                        finance_dict[year_1][line[0]] = line[-2]
        if '合并利润表' in excel and len(excel['合并利润表']) > 0 and verify_header(excel['合并利润表'][0]):
            # year_0, year_1 = excel['合并利润表'][0][2][:4], excel['合并利润表'][0][1][:4]
            for line in excel['合并利润表'][1:]:
                if '基本每股收益' in line[0] or '每股收益' == line[0]:
                    if len(line) == 2:
                        if line[0] not in finance_dict[year] or finance_dict[year][line[0]] == "":
                            finance_dict[year]['每股收益'] = line[-1]
                    else:
                        if line[-1].strip() != '':
                            finance_dict[year_0]['每股收益'] = line[-1]
                        if line[-2].strip() != '':
                            finance_dict[year_1]['每股收益'] = line[-2]
                else:
                    line[0] = re.sub(attribute_fix_pattern, "", line[0].replace(" ","").strip())
                    # line[0] = line[0].replace(" ", "").strip()
                    if line[0] in financial_alias_inv_mapping:
                        line[0] = financial_alias_inv_mapping[line[0]]
                    attribute_list.append(line[0])
                    if len(line) == 2:
                        if line[0] not in finance_dict[year] or finance_dict[year][line[0]] == "":
                            finance_dict[year][line[0]] = line[-1]
                    else:
                        if line[0] not in finance_dict[year_0] or finance_dict[year_0][line[0]] == "":
                            finance_dict[year_0][line[0]] = line[-1]
                        if line[0] not in finance_dict[year_1] or finance_dict[year_1][line[0]] == "":
                            finance_dict[year_1][line[0]] = line[-2]
        # if '股本' in excel and excel['股本'] != None:
        #     assert excel['股本'][1][0] == '期末余额'
        #     attribute_list.append('总股数')
        #     if excel['股本'][1][1] is not None:
        #         finance_dict[year_1]['总股数'] = excel['股本'][1][1].replace(" ","").strip().replace(",", "").replace("，", "")
        #     if excel['股本'][0][1] is not None:
        #         finance_dict[year_0]['总股数'] = excel['股本'][0][1].replace(" ","").strip().replace(",", "").replace("，", "")
        #     if excel['股本'][1][1] is not None and excel['股本'][0][1] is not None:
        #         finance_dict[year_1]['平均总股数'] = str((float(finance_dict[year_1]['总股数']) + float(finance_dict[year_0]['总股数'])) / 2)
        # if '主要会计数据和财务指标' in excel:
            # header_set[tuple(excel['主要会计数据和财务指标'][0][:3])] += 1
        excel['财务报表'] = finance_dict
        return excel

    def process_single_column_excel(excel, year):
        # excel = excel[1:]
        # financial_dict = {}
        # for k, v in excel:
        #     if k not in title_list:
        #         financial_dict[k] = v
        # return financial_dict
        pass

    def load_merged_excels():
        excel_mapping = {}
        file_list = [line.strip().replace(".pdf", ".json") for line in open('data/C-list-pdf-name.txt', 'r', encoding='utf8').readlines()]
        for file in tqdm(file_list, desc="preprocessing excels"):
            filename = os.path.basename(file)
            _, full_name, stock_code, short_name, year, _ = filename.split("__")
            year = year.replace("年", "")
            if os.path.exists(os.path.join('data/processed_excels', file)):
                if args.enable_past_year:
                    excel_mapping[(stock_code, year)] = preprocess_excels(json.load(open(os.path.join('data/processed_excels', file), 'r', encoding='utf8')), year)
                else:
                    financial_dict = process_single_column_excel(json.load(open(os.path.join('data/final_excels', file), 'r', encoding='utf8')), year)
                    excel_mapping[(stock_code, year)] = json.load(open(os.path.join('data/processed_excels', file), 'r', encoding='utf8'))
                    excel_mapping[(stock_code, year)]['财务报表'] = financial_dict

        return excel_mapping

    def load_company_infos():
        company_mapping = {}
        file_list = [line.strip().replace(".pdf", ".json") for line in open('data/C-list-pdf-name.txt', 'r', encoding='utf8').readlines()]
        for file in file_list:
            filename = os.path.basename(file)
            _, full_name, stock_code, short_name, year, _ = filename.split("__")
            year = year.replace("年", "")
            if os.path.exists(os.path.join('data/processed_excels', file)):
                company_mapping[(stock_code, year)] = json.load(open(os.path.join('data/company_info',file), 'r', encoding='utf8'))
        return company_mapping

    excel_mapping = load_merged_excels()
    company_infos = load_company_infos()
    for k in excel_mapping:
        excel_mapping[k]['公司信息'] = company_infos[k]
    new_counter = {}
    old_counter = Counter(attribute_list)
    for k in finance_features:
        if k in old_counter:
            new_counter[k] = old_counter[k]
        else:
            print(k)
    # print(json.dumps(new_counter, ensure_ascii=False, indent=4))
    # print(json.dumps({k:v for k,v in old_counter.items() if v > 20}, ensure_ascii=False, indent=4))
    # print(json.dumps({','.join(k):v for k,v in header_set.items() if v > 20}, ensure_ascii=False, indent=4))
    return excel_mapping


def extract_company_names(extracted_info):
    res = []
    for line in extracted_info:
        if '公司名称' in line:
            res.append(line['公司名称'])
        else:
            res.append('无')
    return res


def query_stock_code(companies):
    extractor = BM25EnityExtractor()
    return extractor.query_company_names(companies)


def set_stock_code(company_infos):
    company_names = extract_company_names(company_infos)
    stock_codes = query_stock_code(company_names)
    for company_info, stock_code in zip(company_infos, stock_codes):
        if '公司名称' in company_info:
            company_info['股票代码'] = stock_code
    return company_infos


def load_frames():
    title_list = ['所有者权益：', '所有者权益', '教育程度', '学历结构类别', '研发人员年龄构成', '专业构成', '研发人员学历结构', '研发人员年龄结构',
    '教育程度类别', '专业构成类别', '', '学历结构类别 学历结构人数', '项目', '备', '列）', '研发人员学历', '非流动负债：',
    '每股收益：', '-', '非流动资产：', '按经营持续性分类', '按所有权归属分类', '总额', '流动资产：']
    brackets_pattern = r'([(（][^)）]*[)）])'
    result = defaultdict(lambda :[])
    counter = defaultdict(int)
    file_list = [line.strip().replace(".pdf", ".json") for line in open('data/C-list-pdf-name.txt', 'r', encoding='utf8').readlines()]
    for file in tqdm(file_list, desc="loading frames"):
        file = os.path.join('data/final_excels', file)
        filename = os.path.basename(file)
        _, full_name, stock_code, short_name, year, _ = filename.split("__")
        sample_dict = {}
        if not os.path.exists(file):
            print(file)
            continue
        for line in json.load(open(file, 'r', encoding='utf8'))[1:]:
            if len(line) != 2:
                continue
            k, v = line
            k = re.sub(brackets_pattern, "", k)
            if len(k) < 2:
                continue
            if k in sample_dict:
                if sample_dict[k] == 0 and v != 0:
                    sample_dict[k] = [v]
            else:
                sample_dict[k] = [v]
                counter[k] += 1
        if len(sample_dict) == 0:
            continue
        sample_dict['公司的中文名称'] = sample_dict['long_company_name']
        sample_dict['公司的中文缩写'] = sample_dict['short_company_name']
        sample_dict.pop('long_company_name')
        sample_dict.pop('short_company_name')
        df = pd.DataFrame(sample_dict, index=[stock_code])
        df = df.drop(list(set(title_list) & set(df.columns)), axis=1)
        result[year.replace("年", "")].append(df)
    filtered_columns = [k for k, v in counter.items() if v >= 200]
    for k, v in result.items():
        for idx, item in enumerate(v):
            v[idx] = item[list(set(item.columns) & set(filtered_columns))]
    # print(result['2019'])
    result_list = []
    for k, v in tqdm(result.items(), desc="merging frames, columns:{}".format(len(filtered_columns))):
        result_list.append(pd.concat(v, axis=0).assign(年份=k))
    # print(result['2019'])
    return pd.concat(result_list, axis=0)

    

def get_response_ask(pipe):
    global excels,frame
    global model, tokenizer
    if model is None:
        model, tokenizer = load_model()
        model.eval()
    while True:
        pipe.send("request_input")  # 请求输入
        question = pipe.recv()  # 等待主进程传入输入
        print("生成分类结果")

        dump_classification_results(refresh=args.refresh_classification,question=question)
        # 提取公司代码
        print("提取公司代码")
        company_infos = set_stock_code(load_extracted())

        print("load_excels")
        
        questions =[]
        questions.append(question)
        finance_answers = []
        complex_answers = []
        personal_answers = []
        company_answers = []
        common_answers = []
        open_answers = []
        if model is None:
            model, tokenizer = load_model()
        for question, company_info in zip(questions, company_infos):
            if len(re.findall(r'(2019|2020|2021)', question)) == 0 and isinstance(company_info, dict) and '类型' in company_info and company_info['类型'] != '常识问题':
                company_info['类型'] = '常识问题'
            if '公司名称' in company_info is not None and company_info['股票代码'] is None:
                print(question, company_info['公司名称'])

        question_type = load_extracted()[0]['类型']
        if question_type == "财务问题":
            print("财务问题")
            finance_querier = finance_table_query(model, tokenizer, excels, finance_features, finance_compute_feaures, args)
            finance_answers = finance_querier.run_query(questions, company_infos, batch_size=16)
            print(finance_answers)
            pipe.send(finance_answers)
            # return finance_answers
        if question_type == "查询问题":
            print("查询问题")
            complex_querier = complex_query(None, tokenizer, excels, frame)
            complex_answers = complex_querier.run_query(questions, company_infos, batch_size=16)
            print(complex_answers)
            pipe.send(complex_answers)
            # return complex_answers
        if question_type == "人员问题":
            print("人员问题")
            personal_querier = personal_query(model, tokenizer, excels)
            personal_answers = personal_querier.run_query(questions, company_infos, batch_size=16)
            print(personal_answers)
            pipe.send(personal_answers)
            # return personal_answers
        if question_type == "公司问题":
            print("公司问题")
            company_querier = company_info_query(model, tokenizer, excels)
            company_answers = company_querier.run_query(questions, company_infos, batch_size=16)
            print(company_answers)
            pipe.send(company_answers)
            # return company_answers
        if question_type == "常识问题":
            print("常识问题")
            common_querier = common_query(model, tokenizer, excels)
            common_answers = common_querier.run_query(questions, company_infos, batch_size=1)
            print(common_answers)
            pipe.send(common_answers)
            # return common_answers
        gc.collect()
        if question_type == "开放问题":
            print("开放问题")
            open_querier = open_query(model, tokenizer, excels)
            open_answers = open_querier.run_query(questions, company_infos, batch_size=1)
            print(open_answers)
            pipe.send(open_answers)
            # return open_answers
        if question_type == "失败":
            # 将模型转移到 CPU
            model = model.to("cpu")

            # 删除模型对象
            del model

            # 运行垃圾回收
            gc.collect()

            # 释放显存
            torch.cuda.empty_cache()
            pipe.send("finish")  # 结束
            break
            # return open_answers



def dump_answers(json_data):
    with open('/tmp/result.json', 'w', encoding='utf8') as fp:
        for item in json_data:
            # assert item is not None
            fp.write(json.dumps(item, ensure_ascii=False) + "\n")




def read_classification_jsonl(file_path='finetune/table_qa/data/classification.jsonl'):
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # 每行是一个 JSON 对象，使用 json.loads 来解析
                data.append(json.loads(line))
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except json.JSONDecodeError:
        print(f"Error decoding JSON in file: {file_path}")
    return data




@app.get("/question/{question}")
async def ask(question: str):
    client_question.value = question
    get_question_event.set()
    get_question_event.clear()
    await asyncio.get_event_loop().run_in_executor(None, get_answer_event.wait)  # 使用 run_in_executor 以避免阻塞
    get_answer_event.clear()  # 重置事件
    data=read_classification_jsonl()
    return {"answer": client_answer.value,"classResult":data}


async def inference_flow():
    global excels,frame
    excels = load_excels()
    frame = load_frames()

    parent_conn, child_conn = mp.Pipe()

    while True:
        p = mp.Process(target=get_response_ask, args=(child_conn,))
        p.start()
        while True:
            msg = parent_conn.recv()
            if msg == "request_input":
                get_question_event.wait()  # 等待问题事件的触发
                parent_conn.send(client_question.value)  # 将输入传递给子进程
            elif msg == "finish":
                p.join()
                break
            else:
                client_answer.value = msg
                get_answer_event.set()  # 触发回答事件

def start_inference_flow():
    asyncio.run(inference_flow())

if __name__ == '__main__':
    server_process = mp.Process(target=uvicorn.run, args=(app,), kwargs={"host": "0.0.0.0", "port": 8000})
    inference_process = mp.Process(target=start_inference_flow)

    server_process.start()
    inference_process.start()

    server_process.join()
    inference_process.join()
