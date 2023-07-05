# CLML-TLLite-Digit-Training

使用TensorflowLite框架扩展数字识别模型，支持训练新字符并识别

将官方案例[digit_classifier](https://github.com/tensorflow/examples/tree/master/lite/examples/digit_classifier/ios)进行扩展，支持新字符训练和识别。训练模型的导出和使用参考另一Demo：[model_personalization](https://github.com/tensorflow/examples/tree/master/lite/examples/model_personalization)(仅有Android的应用)

使用训练的接口时，需要导入[TensorFlowLiteSelectTfOps](https://github.com/tensorflow/examples/tree/master/lite/examples/model_personalization)，否则不支持

模型导出的代码详见：mnist_tflite_trans.ipynb