import sys
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/qikt_switch'))
from compare_validation import ClusterAUC


def test_cluster_auc_matches_sklearn_weighted_auc_with_ties_and_missing_clusters():
    y=np.array([0,1,1,0,1,0,1,0]);p=np.array([.2,.2,.7,.4,.7,.9,.4,.4]);c=np.array([0,0,1,1,2,2,3,3])
    f=ClusterAUC(y,p,c)
    for w in [np.ones(4),np.array([2,0,1,3]),np.array([0,1,0,0])]:
        assert np.isclose(f(w),roc_auc_score(y,p,sample_weight=w[c]),atol=1e-14)
