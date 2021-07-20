import argparse
import json

settingsPath = "settings.json"

# 獲取全部設定
def getAllKeys(args):
    with open(settingsPath, mode = 'r') as file:
        print(file.read())
        # print(json.dumps(json.load(file)))


# 獲取其中一個設定
def getKey(args):
    with open(settingsPath, mode = 'r') as file:
        settings = json.load(file)
        print(str(settings[args.key]))


# 設定
def setKey(args):
    with open(settingsPath, mode = 'w') as file:
        settings = json.load(file)
        settings[args.key] = args.val
        json.dump(settings, file)
        print('Success', str(settings[args.key])
        

def buildArgparser():
    # create the top-level parser
    parser = argparse.ArgumentParser(description='Get or set.')
    subparsers = parser.add_subparsers()

    # create the parser for the "Get all" command
    parser_A = subparsers.add_parser('all', help="Get all keys.")
    parser_A.add_argument('-a', type=str, help='input all keys.', required=True)
    parser_A.set_defaults(func=getAllKeys)


    # create the parser for the "Get" command
    parser_B = subparsers.add_parser('get', help="Get value from key.")
    parser_B.add_argument('-k', type=str, help='input key.', required=True)
    parser_B.set_defaults(func=getKey)


    # create the parser for the "Set" command
    parser_C = subparsers.add_parser('set', help="Set key and value.")
    parser_C.add_argument('-k', type=str, help='input key.', required=True)
    parser_C.add_argument('-v', type=str, help='input value.', required=True)
    parser_C.set_defaults(func=setKey)

    # 解析參數
    args = parser.parse_args()
    args.func(args)

buildArgparser()

