import sys, getopt

from synthetic_data import SyntheticData
from synthetic_data_with_clusters import SyntheticDataWithClusters

def main(argv):

    _, args = getopt.getopt(argv, [])

    num_shifted_genes = int(args[0])
    shifted_mean = int(args[1])

    synthetic_data = SyntheticData(num_shifted_genes, shifted_mean)
    synthetic_data_with_clusters = SyntheticDataWithClusters(num_shifted_genes, shifted_mean)

    print "-------------------------------------------------------------------------------"
    print "-------------------Testing Feed Forward Neural Network----------------------"
    print "-------------------------------------------------------------------------------"

    synthetic_data.test_MLP()

    print "-------------------------------------------------------------------------------"
    print "-------------------Testing Recurrent Neural Network-------------------------"
    print "-------------------------------------------------------------------------------"

    synthetic_data.test_RNN()

    print "-------------------------------------------------------------------------------"
    print "-------------------Testing Superlayered Neural Network----------------------"
    print "-------------------------------------------------------------------------------"

    synthetic_data_with_clusters.test_SNN()

if __name__ =='__main__':
    main(sys.argv[1:])