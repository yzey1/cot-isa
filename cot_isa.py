import argparse
import os
import yaml
import time
from inference import direct_inference, cot_inference, cot_fewshot_inference
from utils import get_data, evaluate_result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--infer_type", type=str, default="cot", choices=['direct', 'cot', 'cot_fewshot'], help="Inference type: direct or cot")
    parser.add_argument("-m", "--model_name", type=str, default="llama3.2", help="Model name")
    parser.add_argument("-d", "--data_fname", type=str, default="Implicit_Labeled_data_for_test.csv", help="Data file name")

    # infer_type = "direct"
    # model_name = "llama3.2"
    # data_fname = "Implicit_Labeled_data_for_test.csv"
    
    # load config
    args = parser.parse_args()
    infer_type = args.infer_type
    model_name = args.model_name
    data_fname = args.data_fname
    config = yaml.load(open("config.yaml", 'r'), Loader=yaml.FullLoader)
    output_path = config['output_path']
    
    # load data
    data = get_data(data_fname)
    print(f"File {data_fname} loaded, data size: {data.shape}")
    
    # inference
    print("Start inference...")
    start_t = time.time()
    preds = []
    reasoning_texts = []
    error_rows = []
    for i, row in data.iterrows():
        try:
            if infer_type == "direct":
                reasoning_text, pred = direct_inference(row['sentence'], row['target'], model_name)
            elif infer_type == "cot":
                reasoning_text, pred = cot_inference(row['sentence'], row['target'], model_name)
            elif infer_type == "cot_fewshot":
                reasoning_text, pred = cot_fewshot_inference(row['sentence'], row['target'], model_name)
        except Exception as e:
            print(f"Error in row {i}: {e}")
            error_rows.append(i)
            reasoning_text, pred = [], 2
        preds.append(pred)
        reasoning_texts.append(reasoning_text)
        
        if i % (data.shape[0] // 10) == 0:
            print(f"Inference in progress: {i} rows done.({i/data.shape[0]*100:.2f}%), time elapsed: {time.time()-start_t:.2f}s")
    
    print("Inference done.")
    
    # save the result
    result = data.copy()
    result['pred'] = preds
    result['reasoning'] = reasoning_texts
    result.to_csv(os.path.join(output_path, f"result_{model_name}_{infer_type}_{data_fname.split('.')[0]}.csv"), index=False)
    
    print(f"Result saved to {output_path}")
    
    # evaluate the result
    acc, f1 = evaluate_result(result)
    acc_explicit, f1_explicit = evaluate_result(result[result['implicit'] == 0])
    acc_implicit, f1_implicit = evaluate_result(result[result['implicit'] == 1])
    
    print(f"Accuracy: {acc:.4f}, F1: {f1:.4f}")
    print(f"Accuracy (explicit): {acc_explicit:.4f}, F1 (explicit): {f1_explicit:.4f}")
    print(f"Accuracy (implicit): {acc_implicit:.4f}, F1 (implicit): {f1_implicit:.4f}")
    print(f"Error rows: {error_rows}")
    
    # save the evaluation result
    with open(os.path.join(output_path, f"evaluation_{model_name}_{infer_type}_{data_fname.split('.')[0]}.txt"), 'w') as f:
        f.write(f"Accuracy: {acc:.4f}, F1: {f1:.4f}\n")
        f.write(f"Accuracy (explicit): {acc_explicit:.4f}, F1 (explicit): {f1_explicit:.4f}\n")
        f.write(f"Accuracy (implicit): {acc_implicit:.4f}, F1 (implicit): {f1_implicit:.4f}\n")
        f.write(f"Error rows: {error_rows}\n")
        
    print(f"Evaluation result saved to {output_path}")
    
    print("Done.")