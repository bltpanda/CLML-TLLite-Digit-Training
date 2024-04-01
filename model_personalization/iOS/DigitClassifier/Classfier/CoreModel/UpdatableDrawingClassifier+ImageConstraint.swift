/*
See LICENSE folder for this sample’s licensing information.

Abstract:
Updatable Drawing Classifier extension with a convenience image constraint property.
*/

import CoreML

/// - Tag: ImageConstraintProperty
extension UpdatableDrawingClassifier {
    /// Returns the image constraint for the model's "drawing" input feature.
    var imageConstraint: MLImageConstraint {
        let description = model.modelDescription
        
        let inputName = "drawing"
        let imageInputDescription = description.inputDescriptionsByName[inputName]!
        
        for (inputName, inputDescription) in description.inputDescriptionsByName {
            print("Input Name: \(inputName)")
            print("Input Type: \(inputDescription.type)")
            print("Input Size: \(inputDescription.multiArrayConstraint?.shape.description ?? "N/A")")
        }
        
        for (outputName, outputDescription) in description.outputDescriptionsByName {
            print("Output Name: \(outputName)")
            print("Output Type: \(outputDescription.type)")
            print("Output Size: \(outputDescription.multiArrayConstraint?.shape.description ?? "N/A")")
        }


        
        return imageInputDescription.imageConstraint!
    }
}
