//
//  CoreModelManager.swift
//  Pixy
//
//  Created by Xander on 2023/11/20.
//

import Foundation
import Photos
import UIKit


struct CoreModelManager {

    static func updateModel(image: UIImage, label: String, callback: @escaping () -> Void) {
        DispatchQueue.global(qos:.background).async {
            var inputSet = CoreModelInputSet.init(for:label)
            inputSet.addDrawing(UserDrawing(image: image.cgImage!))
        
            
            ModelUpdater.shared.updateWith(trainingData: inputSet.featureBatchProvider, completionHandler: callback)
        }
    }
    
    static func predictLabel(of image: UIImage, callback: @escaping (String) -> Void) {
        DispatchQueue.global(qos:.background).async {
            let label = ModelUpdater.shared.predictLabelFor(UserDrawing(image: (image.cgImage!)).featureValue)
            print("predictLabelFor \(String(describing: label))")
            
            DispatchQueue.main.async {
                callback(label ?? "predictLabel error");
            }
        }
    }
    
}


struct UpdateRecord: Identifiable {
    let id: UUID = UUID()
    var albumName: String
    var progress: Double
    var isCompleted: Bool
}
