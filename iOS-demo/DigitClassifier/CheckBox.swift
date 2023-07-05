//
//  CheckBox.swift
//  DigitClassifier
//
//  Created by Xander on 2023/7/4.
//  Copyright © 2023 Google Inc. All rights reserved.
//

import Foundation
import UIKit
class CheckBox: UIButton {
    // Images
    let uncheckedImage = UIImage(systemName: "checkmark.circle")
    let checkedImage = UIImage(systemName: "checkmark.circle.fill")

    override init(frame: CGRect) {
        super.init(frame: frame)
        self.isChecked = false
        self.setImage(uncheckedImage, for: .normal)
        self.setTitleColor(.black, for: .normal)
    }
    
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
    
    // Bool property
    var isChecked: Bool = false {
        didSet{
            if isChecked == true {
                self.setImage(checkedImage, for: .normal)
            } else {
                self.setImage(uncheckedImage, for: .normal)
            }
        }
    }
}
