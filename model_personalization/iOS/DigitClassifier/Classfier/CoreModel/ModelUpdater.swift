/*
See LICENSE folder for this sample’s licensing information.

Abstract:
Manager responsible for updating and using the correct Drawing Classifier at runtime.
*/

import CoreML

/// Class that handles predictions and updating of UpdatableDrawingClassifier model.
class ModelUpdater: ObservableObject{
    static let shared = ModelUpdater()
    
    private var isUpdating = false
    
    struct UpdateRecord: Codable, Identifiable {
        private static let appDirectory = FileManager.default.urls(for: .applicationSupportDirectory,
                                                                   in: .userDomainMask).first!
         let id: UUID
         let time: Date
        
        func filePath() -> String {
            return "\(id).mlmodelc"
        }
        
        func fullPathURL() -> URL {
            return UpdateRecord.appDirectory.appendingPathComponent(filePath())
        }
    }
    
    // 使用日期作为键的字典来存储更新记录
    @Published var updateHistory: [Date: [UpdateRecord]] = [:]
    private var latestUpdate: UpdateRecord?
    
    private init() {
        loadUpdateHistory()
        findLatestUpdate()
        loadUpdatedModel()
    }
    
    private func findLatestUpdate() {
        if let latestDate = updateHistory.keys.sorted().last {
            latestUpdate = updateHistory[latestDate]?.sorted(by: { $0.time > $1.time }).first
        }
    }

//    func updateModel(with data: Data, to newPath: String) {
//        // 将数据保存到新路径的逻辑
//        let record = UpdateRecord(id: UUID(), time: Date())
//        let calendar = Calendar.current
//        let date = calendar.startOfDay(for: record.time)
//            
//        if updateHistory[date] != nil {
//            updateHistory[date]?.append(record)
//        } else {
//            updateHistory[date] = [record]
//        }
//        saveUpdateHistory()
//    }

    private func loadUpdateHistory() {
        if let data = UserDefaults.standard.data(forKey: "updateHistory"),
            let history = try? JSONDecoder().decode([Date: [UpdateRecord]].self, from: data) {
            updateHistory = history
        }
    }

    private func saveUpdateHistory() {
        findLatestUpdate()
        loadUpdatedModel()
        if let data = try? JSONEncoder().encode(updateHistory) {
            UserDefaults.standard.set(data, forKey: "updateHistory")
        }
    }

    func deleteUpdate(recordId: UUID, on date: Date) {
        updateHistory[date]?.removeAll { $0.id == recordId }
        if updateHistory[date]?.isEmpty == true {
            updateHistory.removeValue(forKey: date)
        }
        saveUpdateHistory()
    }
    
    func deleteUpdate(on date: Date) {
        updateHistory.removeValue(forKey: date)
        saveUpdateHistory()
    }
    
    
    // MARK: - Private Type Properties
    /// The updated Drawing Classifier model.
    private var updatedDrawingClassifier: UpdatableDrawingClassifier?
    /// The default Drawing Classifier model.
    private var defaultDrawingClassifier: UpdatableDrawingClassifier {
        do {
            return try UpdatableDrawingClassifier(configuration: .init())
        } catch {
            fatalError("Couldn't load UpdatableDrawingClassifier due to: \(error.localizedDescription)")
        }
    }

    // The Drawing Classifier model currently in use.
    private var liveModel: UpdatableDrawingClassifier {
        updatedDrawingClassifier ?? defaultDrawingClassifier
    }
    
    /// The location of the app's Application Support directory for the user.
//    private let appDirectory = FileManager.default.urls(for: .applicationSupportDirectory,
//                                                               in: .userDomainMask).first!
//    
    /// The default Drawing Classifier model's file URL.
    private let defaultModelURL =  UpdatableDrawingClassifier.urlOfModelInThisBundle
    
//    private static let defaultModelURL =  UpdatableDrawingClassifier.urlOfModelInThisBundle
    /// The permanent location of the updated Drawing Classifier model.
//    private var updatedModelURL = appDirectory.appendingPathComponent("personalized.mlmodelc")
    /// The temporary location of the updated Drawing Classifier model.
//    private var tempUpdatedModelURL = appDirectory.appendingPathComponent("personalized_tmp.mlmodelc")
    
    /// Triggers code on the first prediction, to (potentially) load a previously saved updated model just-in-time.
//    private static var hasMadeFirstPrediction = false
    

    // MARK: - Public Properties
    var imageConstraint: MLImageConstraint {
        liveModel.imageConstraint
    }
    
    // MARK: - Public Type Methods
    func predictLabelFor(_ value: MLFeatureValue) -> String? {
//        if !hasMadeFirstPrediction {
//            hasMadeFirstPrediction = true
//            
//            // Load the updated model the app saved on an earlier run, if available.
//            loadUpdatedModel()
//        }
        
        return liveModel.predictLabelFor(value)
    }
 
    
    /// Updates the model to recognize images simlar to the given drawings contained within the `inputBatchProvider`.
    /// - Parameters:
    ///     - trainingData: A collection of sample images, each paired with the same label.
    ///     - completionHandler: The completion handler provided from a view controller.
    /// - Tag: CreateUpdateTask
    func updateWith(trainingData: MLBatchProvider,
                           completionHandler: @escaping () -> Void) {
        
        
        
        let currentModelURL = latestUpdate?.fullPathURL() ?? defaultModelURL

        
        /// The closure an MLUpdateTask calls when it finishes updating the model.
        func updateModelCompletionHandler(updateContext: MLUpdateContext) {
            // Save the updated model to the file system.
            saveUpdatedModel(updateContext)
            
            // Begin using the saved updated model.
            loadUpdatedModel()
            
            // Inform the calling View Controller when the update is complete
            DispatchQueue.main.async { completionHandler() }
        }
        
        print("update start")
        UpdatableDrawingClassifier.updateModel(at: currentModelURL,
                                               with: trainingData,
                                               completionHandler: updateModelCompletionHandler)
    }
    
    func updateWith(trainingData: MLBatchProvider) async {
            let currentModelURL = latestUpdate?.fullPathURL() ?? defaultModelURL

            await withCheckedContinuation { continuation in
                
                print("update start")
                UpdatableDrawingClassifier.updateModel(at: currentModelURL, with: trainingData) { updateContext in
                    // Save the updated model to the file system.
                    self.saveUpdatedModel(updateContext)

                    // Begin using the saved updated model.
                    self.loadUpdatedModel()
                    
                    print("update close")
                    // Resume the continuation after the update is complete.
                    continuation.resume()
                }
            }
        }
    
    /// Deletes the updated model and reverts back to original Drawing Classifier.
//    func resetDrawingClassifier() {
//        // Clear the updated Drawing Classifier.
//        updatedDrawingClassifier = nil
//        
//        // Remove the updated model from its designated path.
//        if FileManager.default.fileExists(atPath: updatedModelURL.path) {
//            try? FileManager.default.removeItem(at: updatedModelURL)
//        }
//    }
    
    // MARK: - Private Type Helper Methods
    /// Saves the model in the given Update Context provided by an MLUpdateTask.
    /// - Parameter updateContext: The context from the Update Task that contains the updated model.
    /// - Tag: SaveUpdatedModel
    private func saveUpdatedModel(_ updateContext: MLUpdateContext) {
        let updatedModel = updateContext.model
        let fileManager = FileManager.default
        do {
            let uuid = UUID()
            let record = UpdateRecord(id: uuid, time: Date())
            let savedPathURL = record.fullPathURL()
            // Create a directory for the updated model.
            try fileManager.createDirectory(at: savedPathURL,
                                            withIntermediateDirectories: true,
                                            attributes: nil)
            
            // Save the updated model to temporary filename.
            try updatedModel.write(to: savedPathURL)
            
            // Replace any previously updated model with this one.
//            _ = try fileManager.replaceItemAt(updatedModelURL,
//                                              withItemAt: tempUpdatedModelURL)
            let calendar = Calendar.current
            let date = calendar.startOfDay(for: record.time)
                
            if updateHistory[date] != nil {
                updateHistory[date]?.append(record)
            } else {
                updateHistory[date] = [record]
            }
            saveUpdateHistory()
            
            print("Updated model saved to:\n\t\(savedPathURL)")
        } catch let error {
            print("Could not save updated model to the file system: \(error)")
            return
        }
    }
    
    /// Loads the updated Drawing Classifier, if available.
    /// - Tag: LoadUpdatedModel
    private func loadUpdatedModel() {
        updatedDrawingClassifier = nil
        if let filePathURL = latestUpdate?.fullPathURL() {
            guard FileManager.default.fileExists(atPath: filePathURL.path) else {
                // The updated model is not present at its designated path.
                return
            }
            
            // Create an instance of the updated model.
            guard let model = try? UpdatableDrawingClassifier(contentsOf: filePathURL) else {
                return
            }
            
            // Use this updated model to make predictions in the future.
            updatedDrawingClassifier = model
        }
    }
}
