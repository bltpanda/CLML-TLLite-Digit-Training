// Copyright 2019 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import UIKit
import Sketch

class ViewController: UIViewController, SketchViewDelegate {

  @IBOutlet weak var resultLabel: UILabel!
  @IBOutlet weak var sketchView: SketchView!
  private var classifier: DigitClassifier?
    private var trainCheckBox: CheckBox = CheckBox()
    private var inferCheckBox: CheckBox = CheckBox()

    private var selectLabelIndex: Int = -1
    private var textField: UITextField?

  override func viewDidLoad() {
    super.viewDidLoad()

    // Setup sketch view.
    sketchView.lineWidth = 30
    sketchView.backgroundColor = UIColor.black
    sketchView.lineColor = UIColor.white
    sketchView.sketchViewDelegate = self
      
      
      resultLabel.superview?.addSubview(trainCheckBox)
      trainCheckBox.frame = CGRect(x: 10, y: resultLabel.frame.origin.y + 80, width: 80, height: 40)
      trainCheckBox.setTitle("Train", for: .normal)
      trainCheckBox.addTarget(self, action: #selector(checkBoxClicked), for: .touchUpInside)
      
      
      resultLabel.superview?.addSubview(inferCheckBox)
      inferCheckBox.frame = CGRect(x: 100, y: resultLabel.frame.origin.y + 80, width: 80, height: 40)
      inferCheckBox.setTitle("Infer", for: .normal)
      inferCheckBox.isChecked = true
      inferCheckBox.addTarget(self, action: #selector(checkBoxClicked), for: .touchUpInside)
      
      
      let fullScreenSize = UIScreen.main.bounds.size
    
      // 建立一個 UITextField
      textField = UITextField(frame: CGRect(
        x: 0, y: resultLabel.frame.origin.y + 130,
        width: fullScreenSize.width, height: 40))

      // 建立 UIPickerView
      let myPickerView = UIPickerView()

      // 設定 UIPickerView 的 delegate 及 dataSource
      myPickerView.delegate = self
      myPickerView.dataSource = self

      // 將 UITextField 原先鍵盤的視圖更換成 UIPickerView
      textField!.inputView = myPickerView
      textField?.placeholder = "Please select the training label"


      // 設置 UITextField 其他資訊並放入畫面中
      textField!.backgroundColor = UIColor.init(
        red: 0.9, green: 0.9, blue: 0.9, alpha: 1)
      textField!.textAlignment = .center
      resultLabel.superview?.addSubview(textField!)
      
      textField?.isHidden = !self.trainCheckBox.isChecked
      
    // Initialize a DigitClassifier instance
    DigitClassifier.newInstance { result in
      switch result {
      case let .success(classifier):
        self.classifier = classifier
      case .error(_):
        self.resultLabel.text = "Failed to initialize."
      }
    }
  }

    func makeSelectButtonTitle() -> String {
        if let label = transToLabel(from: selectLabelIndex) {
            return "  Train Label: \(label)"
        } else {
            return "  Please select the training label"
        }
    }
  /// Clear drawing canvas and result text when tapping Clear button.
  @IBAction func tapClear(_ sender: Any) {
    sketchView.clear()
    resultLabel.text = "Please draw a digit."
  }
    
    @objc func checkBoxClicked(_ sender: Any) {
        guard let sender = sender as? CheckBox else {
            return
        }
        if sender.isChecked == false {
            self.inferCheckBox.isChecked = !self.inferCheckBox.isChecked
            self.trainCheckBox.isChecked = !self.trainCheckBox.isChecked
            textField?.isHidden = !self.trainCheckBox.isChecked
        }
    }

  /// Callback executed every time there is a new drawing
  func drawView(_ view: SketchView, didEndDrawUsingTool tool: AnyObject) {
      if trainCheckBox.isChecked {
          classsifyTrain()
      } else {
          classifyDrawing()
      }
  }

  /// Classify the drawing currently on the canvas and display result.
  private func classifyDrawing() {
    guard let classifier = self.classifier else { return }

    // Capture drawing to RGB file.
    UIGraphicsBeginImageContext(sketchView.frame.size)
    sketchView.layer.render(in: UIGraphicsGetCurrentContext()!)
    let drawing = UIGraphicsGetImageFromCurrentImageContext()
    UIGraphicsEndImageContext();

    guard drawing != nil else {
      resultLabel.text = "Invalid drawing."
      return
    }

    // Run digit classifier.
    classifier.classify(image: drawing!) { result in
      // Show the classification result on screen.
      switch result {
      case let .success(classificationResult):
        self.resultLabel.text = classificationResult
      case .error(_):
        self.resultLabel.text = "Failed to classify drawing."
      }
    }
  }
    
    private func classsifyTrain() {
      guard let classifier = self.classifier else { return }

      // Capture drawing to RGB file.
      UIGraphicsBeginImageContext(sketchView.frame.size)
      sketchView.layer.render(in: UIGraphicsGetCurrentContext()!)
      let drawing = UIGraphicsGetImageFromCurrentImageContext()
      UIGraphicsEndImageContext();

      guard drawing != nil else {
        resultLabel.text = "Invalid drawing."
        return
      }

      // Run digit classifier.
      classifier.train(image: drawing!, labelIndex: selectLabelIndex) { result in
        // Show the classification result on screen.
        switch result {
        case let .success(classificationResult):
          self.resultLabel.text = classificationResult
        case .error(_):
          self.resultLabel.text = "Failed to classify drawing."
        }
      }
    }

}


extension ViewController: UIPickerViewDelegate, UIPickerViewDataSource  {
    func numberOfComponents(in pickerView: UIPickerView) -> Int {
        return 1
    }
    
    // UIPickerViewDataSource 必須實作的方法：
    // UIPickerView 各列有多少行資料
    func pickerView(
        _ pickerView: UIPickerView,
        numberOfRowsInComponent component: Int) -> Int {
            // 返回陣列 meals 的成員數量
            return 11
        }
    
    // UIPickerView 每個選項顯示的資料
    func pickerView(_ pickerView: UIPickerView,
                    titleForRow row: Int,
                    forComponent component: Int) -> String? {
        // 設置為陣列 meals 的第 row 項資料
        return transToLabel(from: row)
    }
    
    // UIPickerView 改變選擇後執行的動作
    func pickerView(_ pickerView: UIPickerView,
                    didSelectRow row: Int, inComponent component: Int) {
        
        // 將 UITextField 的值更新為陣列 meals 的第 row 項資料
        selectLabelIndex = row
        textField?.text = transToLabel(from: row)
        textField?.resignFirstResponder()
    }
}
