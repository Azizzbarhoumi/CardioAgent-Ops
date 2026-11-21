# Golden snapshot vs live API (v3)

39/40 OK

| WAV | model | expected label / score / threshold | API got | result |
|---|---|---|---|---|
| pascal_extra_201104021355.wav | svm_linear | 0 / -0.569887 / 0.0000 | 0 / -0.569887 / 0.0000 | OK |
| pascal_extra_201104021355.wav | svm_rbf | 1 / 0.859761 / 0.0000 | 1 / 0.859761 / 0.0000 | OK |
| pascal_extra_201104021355.wav | decision_tree | 1 / 0.693796 / 0.2982 | 1 / 0.693796 / 0.2982 | OK |
| pascal_extra_201104021355.wav | cnn_custom | 1 / 0.495137 / 0.2874 | 1 / 0.495137 / 0.2874 | OK |
| pascal_extra_201104021355.wav | vgg16_transfer | 1 / 0.659979 / 0.4965 | 1 / 0.659979 / 0.4965 | OK |
| pascal_murmur_197_1308141235553_D.wav | svm_linear | 1 / 1.102257 / 0.0000 | 1 / 1.102257 / 0.0000 | OK |
| pascal_murmur_197_1308141235553_D.wav | svm_rbf | 1 / 1.031969 / 0.0000 | 1 / 1.031969 / 0.0000 | OK |
| pascal_murmur_197_1308141235553_D.wav | decision_tree | 1 / 1.000000 / 0.2982 | 1 / 1.000000 / 0.2982 | OK |
| pascal_murmur_197_1308141235553_D.wav | cnn_custom | 1 / 0.827523 / 0.2874 | 1 / 0.827517 / 0.2874 | OK |
| pascal_murmur_197_1308141235553_D.wav | vgg16_transfer | 1 / 0.663217 / 0.4965 | 1 / 0.663187 / 0.4965 | OK |
| pascal_normal_246_1309199278902_D.wav | svm_linear | 1 / 0.814351 / 0.0000 | 1 / 0.814351 / 0.0000 | OK |
| pascal_normal_246_1309199278902_D.wav | svm_rbf | 1 / 0.500237 / 0.0000 | 1 / 0.500237 / 0.0000 | OK |
| pascal_normal_246_1309199278902_D.wav | decision_tree | 1 / 0.693796 / 0.2982 | 1 / 0.693796 / 0.2982 | OK |
| pascal_normal_246_1309199278902_D.wav | cnn_custom | 1 / 0.762187 / 0.2874 | 1 / 0.762153 / 0.2874 | OK |
| pascal_normal_246_1309199278902_D.wav | vgg16_transfer | 1 / 0.683548 / 0.4965 | 1 / 0.683602 / 0.4965 | OK |
| pascal_normal_296_1311682952647_A1.wav | svm_linear | 1 / 0.516091 / 0.0000 | 1 / 0.516088 / 0.0000 | FAIL |
| pascal_normal_296_1311682952647_A1.wav | svm_rbf | 1 / 0.429205 / 0.0000 | 1 / 0.429205 / 0.0000 | OK |
| pascal_normal_296_1311682952647_A1.wav | decision_tree | 1 / 0.459364 / 0.2982 | 1 / 0.459364 / 0.2982 | OK |
| pascal_normal_296_1311682952647_A1.wav | cnn_custom | 1 / 0.323972 / 0.2874 | 1 / 0.323972 / 0.2874 | OK |
| pascal_normal_296_1311682952647_A1.wav | vgg16_transfer | 0 / 0.431283 / 0.4965 | 0 / 0.431283 / 0.4965 | OK |
| physio_training-b_b0001.wav | svm_linear | 1 / 0.010998 / 0.0000 | 1 / 0.010998 / 0.0000 | OK |
| physio_training-b_b0001.wav | svm_rbf | 1 / 0.215858 / 0.0000 | 1 / 0.215858 / 0.0000 | OK |
| physio_training-b_b0001.wav | decision_tree | 0 / 0.000000 / 0.2982 | 0 / 0.000000 / 0.2982 | OK |
| physio_training-b_b0001.wav | cnn_custom | 1 / 0.466319 / 0.2874 | 1 / 0.466319 / 0.2874 | OK |
| physio_training-b_b0001.wav | vgg16_transfer | 1 / 0.645514 / 0.4965 | 1 / 0.645514 / 0.4965 | OK |
| physio_training-b_b0455.wav | svm_linear | 1 / 1.020332 / 0.0000 | 1 / 1.020332 / 0.0000 | OK |
| physio_training-b_b0455.wav | svm_rbf | 1 / 0.193821 / 0.0000 | 1 / 0.193821 / 0.0000 | OK |
| physio_training-b_b0455.wav | decision_tree | 1 / 1.000000 / 0.2982 | 1 / 1.000000 / 0.2982 | OK |
| physio_training-b_b0455.wav | cnn_custom | 1 / 0.308721 / 0.2874 | 1 / 0.308721 / 0.2874 | OK |
| physio_training-b_b0455.wav | vgg16_transfer | 0 / 0.440771 / 0.4965 | 0 / 0.440770 / 0.4965 | OK |
| physio_training-c_c0015.wav | svm_linear | 1 / 3.551678 / 0.0000 | 1 / 3.551678 / 0.0000 | OK |
| physio_training-c_c0015.wav | svm_rbf | 1 / 0.365566 / 0.0000 | 1 / 0.365566 / 0.0000 | OK |
| physio_training-c_c0015.wav | decision_tree | 1 / 1.000000 / 0.2982 | 1 / 1.000000 / 0.2982 | OK |
| physio_training-c_c0015.wav | cnn_custom | 1 / 0.789226 / 0.2874 | 1 / 0.789226 / 0.2874 | OK |
| physio_training-c_c0015.wav | vgg16_transfer | 1 / 0.883021 / 0.4965 | 1 / 0.883022 / 0.4965 | OK |
| physio_training-e_e01899.wav | svm_linear | 1 / 0.158455 / 0.0000 | 1 / 0.158455 / 0.0000 | OK |
| physio_training-e_e01899.wav | svm_rbf | 0 / -0.768155 / 0.0000 | 0 / -0.768155 / 0.0000 | OK |
| physio_training-e_e01899.wav | decision_tree | 0 / 0.000000 / 0.2982 | 0 / 0.000000 / 0.2982 | OK |
| physio_training-e_e01899.wav | cnn_custom | 0 / 0.065012 / 0.2874 | 0 / 0.065012 / 0.2874 | OK |
| physio_training-e_e01899.wav | vgg16_transfer | 0 / 0.361574 / 0.4965 | 0 / 0.361574 / 0.4965 | OK |
