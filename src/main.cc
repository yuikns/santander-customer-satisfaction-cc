/**
 * The MIT License (MIT)
 *
 * Copyright (c) 2015 Yu Jing <yu@argcv.com>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 **/
#include <cstdio>
#include <cstdlib>

#include <string>
#include <vector>

#include "knn.h"

#include "argcv/ml/ml.hh"
#include "argcv/ml/perceptron.hh"
#include "argcv/ml/sgd.hh"
#include "argcv/ml/svm.hh"
#include "argcv/string/string.hh"

using namespace argcv::ml;
using namespace argcv::string;

void pred_print(svm& ctl, const std::string& test, const std::string& result) {
    printf("predict: [%s], [%s]\n", test.c_str(), result.c_str());
    FILE* ftest = fopen(test.c_str(), "r");
    FILE* fresult = fopen(result.c_str(), "w");
    if (ftest != nullptr && fresult != nullptr) {
        fprintf(fresult, "ID,TARGET\n");
        char buff[10000];
        buff[9999] = '\0';
        size_t cnt = 0;
        fgets(buff, 9999, ftest);
        memset(buff, 0, 10000);
        while (fgets(buff, 9999, ftest)) {
            cnt++;
            if (cnt % 1000 == 0) printf("%zu ...", cnt);
            std::vector<std::string> sparams = split(buff, ",");
            std::vector<double> dparams;
            for (size_t ix = 1; ix < sparams.size(); ix++) {
                dparams.push_back(atof(sparams[ix].c_str()));
            }
            fprintf(fresult, "%s,%d", sparams[0].c_str(), ctl.predict(dparams) ? 1 : 0);
            memset(buff, 0, 10000);
        }
        printf("\nall done, size: %zu\n", cnt);
        fclose(ftest);
        fclose(fresult);
    } else {
        if (ftest == nullptr) {
            printf("%s open failed\n", test.c_str());
        } else {
            fclose(ftest);
        }
        if (fresult == nullptr) {
            printf("%s open failed\n", result.c_str());
        } else {
            fclose(fresult);
        }
    }
}

int main(int argc, char* argv[]) {
    const std::string path_prefix = "../../data/";
    const std::string path_train = path_prefix + "train.csv";
    const std::string path_test = path_prefix + "test.csv";
    const std::string path_result = "my_submission.csv";
    // sgd ctl(0.0000000000000000001, 0.001, 10000, 100);
    svm ctl;
    // perceptron ctl(0, 0.001, 300);
    FILE* ftrain = fopen(path_train.c_str(), "r");
    char buff[10000];
    buff[9999] = '\0';
    size_t cnt = 0;
    fgets(buff, 9999, ftrain);
    printf("loading ...");
    memset(buff, 0, 10000);
    while (fgets(buff, 9999, ftrain)) {
        cnt++;
        if (cnt % 1000 == 0) printf("%zu ...", cnt);
        std::vector<std::string> sparams = split(buff, ",");
        std::vector<double> dparams;
        for (size_t ix = 1; ix < sparams.size() - 1; ix++) {
            dparams.push_back(atof(sparams[ix].c_str()));
        }
        double y = atoi(sparams[sparams.size() - 1].c_str()) == 1 ? 1 : -1;
        if (dparams.size() != 369) {
            printf("error: %s -- %lu \n", buff, dparams.size());
        }
        ctl.add(dparams, y > 0);
        memset(buff, 0, 10000);
    }
    fclose(ftrain);
    printf("\n");
    ctl.learn();
    pred_print(ctl, path_test, path_result);
    return 0;
}
