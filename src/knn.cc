#include "knn.h"
#include <math.h>
#include <stdlib.h>

//#include <stdio.h>

#define KNNABS(a, b) (a > b ? a - b : b - a)

/**
 * distances
 */
// p == 2
double knn_euclidean_distance(_kdt_dataset d1, _kdt_dataset d2, int len) {
    double sum = 0;
    int i;
    for (i = 0; i < len; i++) {
        double rs = KNNABS(d1->data[i], d2->data[i]);
        sum += rs * rs;
    }
    return sqrt(sum);
}

// p == 1
double knn_manhattan_distance(_kdt_dataset d1, _kdt_dataset d2, int len) {
    double sum = 0;
    int i;
    for (i = 0; i < len; i++) {
        sum += KNNABS(d1->data[i], d2->data[i]);
    }
    return sum;
}

int kdt_build_cmp(const void *o1, const void *o2) {
    if (*((*(_kdt_dataset *)o1)->dtmp) == *((*(_kdt_dataset *)o2)->dtmp))
        return 0;
    else
        return (*((*(_kdt_dataset *)o1)->dtmp) > *((*(_kdt_dataset *)o2)->dtmp));
}

// 0 1 2 size 3
//     2
// inline void kdt_check_middle_number(_kdt_dataset * dplist,int len,int dept , int * lsize ,int * rsize,int *
// offset)
inline void kdt_check_middle_number(_kdt_dataset *dplist, int len, int *lsize, int *rsize, int *offset) {
    *offset = len / 2;
    // in fact , dplist[(*offset) - 1]'s value  will always less or equals dplist[(*offset)]'s
    while (*offset > 0 && *dplist[(*offset) - 1]->dtmp >= *dplist[(*offset)]->dtmp) {
        (*offset)--;
    }
    *lsize = *offset;
    *rsize = len - *offset - 1;
}

/**
 * generate kd tree
 * make all same data right
 */
_kdtnode kdt_gen(_kdt_dataset *dplist, int len, int dept, _kdtnode pnode) {
    if (len == 0) return NULL;
    int cmp_offset = dept % dplist[0]->len;
    _kdtnode knode = (_kdtnode)malloc(sizeof(kdtnode));
    knode->pa = pnode;
    int i;
    for (i = 0; i < len; i++) {
        dplist[i]->dtmp = &((dplist[i])->data[cmp_offset]);
    }
    qsort(dplist, len, sizeof(_kdt_dataset), kdt_build_cmp);
    int offset;
    int lsize;
    int rsize;
    kdt_check_middle_number(dplist, len, &lsize, &rsize, &offset);
    _kdtnode lsNode = kdt_gen(dplist, lsize, dept + 1, knode);
    _kdtnode rsNode = kdt_gen(dplist + offset + 1, rsize, dept + 1, knode);

    knode->ls = lsNode;
    knode->rs = rsNode;

    knode->data = dplist[offset];

    return knode;
}

// TODO
_kdtnode kdt_init(_kdt_dataset dlist, int setLen) {
    _kdt_dataset *dp = (_kdt_dataset *)malloc(sizeof(_kdt_dataset) * setLen);
    int i = 0;
    for (i = 0; i < setLen; i++) {
        dp[i] = (dlist + i);
    }
    _kdtnode root = kdt_gen(dp, setLen, 0, NULL);
    free(dp);
    return root;
}

inline _kdtnode kdt_get_leaf(_kdtnode node, _kdt_dataset dquery, int dept) {
    int cmp_offset = dept % node->data->len;
    if (dquery->data[cmp_offset] >= node->data->data[cmp_offset]) {
        if (node->rs == NULL)
            return node;
        else
            return kdt_get_leaf(node->rs, dquery, dept + 1);
    } else {
        if (node->ls == NULL)
            return node;
        else
            return kdt_get_leaf(node->ls, dquery, dept + 1);
    }
}

_kdt_dataset kdt_search(_kdtnode kdt_root, _kdt_dataset dquery) {
    _kdtnode ds = kdt_get_leaf(kdt_root, dquery, 0);
    _kdt_dataset tmpds = ds->data;
    int min_kval = knn_euclidean_distance(ds->data, dquery, ds->data->len);
    while (ds->pa != NULL) {
        ds = ds->pa;
        int tmp_kval = knn_euclidean_distance(ds->data, dquery, ds->data->len);
        if (tmp_kval < min_kval) {
            tmpds = ds->data;
        }
    }
    return tmpds;
}

void kdt_destory(_kdtnode root) {
    if (root == NULL) return;

    if (root->ls != NULL) {
        // printf("destory ls\n");
        kdt_destory(root->ls);
    }

    if (root->rs != NULL) {
        // printf("destory rs\n");
        kdt_destory(root->rs);
    }
    free(root);
}
