#ifndef __KNN_H__
#define __KNN_H__

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    double data[1024];
    double *dtmp;
    unsigned int len;
} kdt_dataset, *_kdt_dataset;

typedef struct kdtnode {
    struct kdtnode *pa;  // parent
    struct kdtnode *ls;  // left son
    struct kdtnode *rs;  // right son
    _kdt_dataset data;
} kdtnode, *_kdtnode;

double knn_euclidean_distance(_kdt_dataset d1, _kdt_dataset d2, int len);
double knn_manhattan_distance(_kdt_dataset d1, _kdt_dataset d2, int len);

_kdtnode kdt_init(_kdt_dataset dlist, int setLen);

_kdt_dataset kdt_search(_kdtnode kdt_root, _kdt_dataset dquery);

void kdt_destory(_kdtnode root);

#ifdef __cplusplus
}
#endif

#endif  // __KNN_H__
