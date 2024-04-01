/*
See LICENSE folder for this sample’s licensing information.

Abstract:
The user's drawing, stored as an image, and its location on a PKCanvasView.
*/

import CoreML
import CoreImage

/// Convenience structure that stores a drawing's `CGImage`
/// along with the `CGRect` in which it was drawn on the `PKCanvasView`
/// - Tag: Drawing
struct UserDrawing {
    /// The underlying image of the drawing.
    let image: CGImage
        
    /// Wraps the underlying image in a feature value.
    /// - Tag: ImageFeatureValue
    var featureValue: MLFeatureValue {
        // Get the model's image constraints.
        let imageConstraint = ModelUpdater.shared.imageConstraint
        
        let imageFeatureValue = try? MLFeatureValue(cgImage: image,
                                                    constraint: imageConstraint)
        return imageFeatureValue!
    }
}
