import os
import argparse
import json

from data import genTurnData_nbest

cur_dir = os.path.dirname(os.path.abspath(__file__))


def main():
    parser = argparse.ArgumentParser(description='Simple hand-crafted dialog state tracker baseline.')
    parser.add_argument('--dst_history_dir', dest='dst_history_dir', action='store', default="dst_history",
                        help='The dir of custom dst_history.')
    parser.add_argument('--start_ind', dest='start_ind', action='store', type=int, default=0,
                        help='start index.')
    parser.add_argument('--end_ind', dest='end_ind', action='store', type=int, default=2999,
                        help='end index.')
    parser.add_argument('--output_name', dest='output_name', action='store', default="train_custom.json",
                        help='The output filename.')
    args = parser.parse_args()

    log_list = [os.path.join(cur_dir, args.dst_history_dir, 'log-%d.json'%i)
                for i in xrange(args.start_ind, args.end_ind+1)]
    label_list = [os.path.join(cur_dir, args.dst_history_dir, 'label-%d.json'%i)
                  for i in xrange(args.start_ind, args.end_ind+1)]

    data = []
    for i in xrange(len(log_list)):
        fileData = dict()
        fileData["turns"] = list()
        with open(log_list[i], 'r') as log_file:
            with open(label_list[i], 'r') as label_file:
                log_json = json.load(log_file)
                label_json = json.load(label_file)
                for i in xrange(len(log_json["turns"])):
                    turnData = genTurnData_nbest(log_json["turns"][i], label_json["turns"][i])
                    fileData["turns"].append(turnData)
        data.append(fileData)
    with open(args.output_name, "w") as fw:
        fw.write(json.dumps(data, indent=1))

if __name__ == '__main__':
    main()
