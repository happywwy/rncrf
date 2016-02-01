#from classify.learn_classifiers import evaluate
import argparse
import evaluation_depnn as eval


if __name__ == '__main__':
    
    
    parser = argparse.ArgumentParser(description='QANTA evaluation')
    parser.add_argument('-data', help='location of trees', default='util/data_semEval/final_input_restest_5class_new1')
    parser.add_argument('-model', help='location of trained model', default='models/trainingRes300_params_5class_punkt_new')
    parser.add_argument('-d', help='word embedding dimension', type=int, default=300)
    parser.add_argument('-len', help='training vector length', default = 50)
    parser.add_argument('-c', help='number of classes', type=int, default=5)
    parser.add_argument('-op', help='use mixed word vector or not', default = False)
    
    args = vars(parser.parse_args())
    
    print 'DT-RNN performance: '

    if args['op']:
        eval.evaluate(args['data'], args['model'], args['d'] + args['len'], args['c'])
        #eval.evaluate(args['trees'], args['data'], args['model'], args['d'] + args['len'], args['c'])
    else:
        eval.evaluate(args['data'], args['model'], args['d'], args['c'])
        #eval.evaluate(args['trees'], args['data'], args['model'], args['d'], args['c'])
