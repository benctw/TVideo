import argparse

def getKey(args):
    pass

    
def setKey(args):
    pass


def buildArgparser():
    # create the top-level parser
    parser = argparse.ArgumentParser(description='Get or set.')
    subparsers = parser.add_subparsers()


    # create the parser for the "Get" command
    parser_A = subparsers.add_parser('get', help="Get value from key")
    parser_A.add_argument('-k', type=str, help='input key.', required=True)
    parser_A.set_defaults(func=getKey)


    # create the parser for the "Set" command
    parser_B = subparsers.add_parser('set', help="Set key and value")
    parser_B.add_argument('-k', type=str, help='input key.', required=True)
    parser_B.add_argument('-v', type=str, help='input value.', required=True)
    parser_B.set_defaults(func=setKey)

    # 解析參數
    args = parser.parse_args()
    args.func(args)


buildArgparser()