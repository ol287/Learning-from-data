from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

class DecisionTreePruning:
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        """
        Initialize the Decision Tree with pre-pruning parameters.
        
        Parameters:
        max_depth (int): Maximum depth of the tree.
        min_samples_split (int): Minimum samples required to split a node.
        min_samples_leaf (int): Minimum samples required to be at a leaf node.
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.model = None  # Placeholder for the decision tree model

    def train(self, X_train, y_train):
        """
        Train the decision tree model with specified pre-pruning parameters.
        
        Parameters:
        X_train (array-like): Training features.
        y_train (array-like): Training labels.
        """
        self.model = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf
        )
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model performance using test data.
        
        Parameters:
        X_test (array-like): Test features.
        y_test (array-like): Test labels.
        
        Returns:
        accuracy (float): Model accuracy on the test set.
        confusion (array-like): Confusion matrix for the test set.
        """
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        confusion = confusion_matrix(y_test, y_pred)
        return accuracy, confusion

    def plot_tree(self, feature_names=None):
        """
        Plot the trained decision tree.
        
        Parameters:
        feature_names (list, optional): List of feature names for the plot. 
                                        If not provided, defaults to generic feature names.
        """
        if feature_names is None:
            feature_names = ["Feature"+str(i) for i in range(self.model.tree_.n_features)]
        
        plt.figure(figsize=(20, 10))
        plot_tree(self.model, filled=True, feature_names=feature_names)
        plt.show()


    def post_prune(self, X_train, y_train, X_test, y_test):
        """
        Post-prune the decision tree using cost complexity pruning.
        
        Parameters:
        X_train (array-like): Training features.
        y_train (array-like): Training labels.
        X_test (array-like): Test features.
        y_test (array-like): Test labels.
        
        Returns:
        pruned_model (DecisionTreeClassifier): The pruned decision tree.
        alpha_best (float): The best alpha value for pruning.
        """
        # Get effective alphas and pruned subtrees
        path = self.model.cost_complexity_pruning_path(X_train, y_train)
        ccp_alphas, impurities = path.ccp_alphas, path.impurities
        
        # Store results for the pruned trees
        pruned_models = []
        test_scores = []
        
        # Train a sequence of trees with different values of alpha
        for ccp_alpha in ccp_alphas:
            clf = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                ccp_alpha=ccp_alpha
            )
            clf.fit(X_train, y_train)
            pruned_models.append(clf)
            # Test the model on the test set
            test_scores.append(accuracy_score(y_test, clf.predict(X_test)))

        # Find the alpha with the highest accuracy on the test set
        alpha_best = ccp_alphas[np.argmax(test_scores)]
        pruned_model = pruned_models[np.argmax(test_scores)]
        
        # Plot accuracy vs alpha to find the optimal pruning point
        plt.figure(figsize=(10, 6))
        plt.plot(ccp_alphas, test_scores, marker="o", drawstyle="steps-post")
        plt.xlabel("Alpha")
        plt.ylabel("Accuracy")
        plt.title("Accuracy vs Alpha for Post-Pruning")
        plt.show()

        # Update the model to the pruned model with best alpha
        self.model = pruned_model
        return pruned_model, alpha_best

# Example usage
if __name__ == "__main__":
    from sklearn.datasets import load_iris

    # Load data and split into train/test
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3, random_state=42)

    # Initialize the DecisionTreePruning with pre-pruning parameters
    tree_model = DecisionTreePruning(max_depth=4, min_samples_split=5, min_samples_leaf=2)
    
    # Train the model
    tree_model.train(X_train, y_train)
    
    # Evaluate the model on test set
    accuracy, confusion = tree_model.evaluate(X_test, y_test)
    print("Initial Test Accuracy:", accuracy)
    print("Initial Confusion Matrix:\n", confusion)
    
    # Plot the initial decision tree
    tree_model.plot_tree()
    
    # Post-prune the model and get the pruned model with optimal alpha
    pruned_model, alpha_best = tree_model.post_prune(X_train, y_train, X_test, y_test)
    print("Best Alpha for Pruning:", alpha_best)
    
    # Evaluate the pruned model on the test set
    pruned_accuracy, pruned_confusion = tree_model.evaluate(X_test, y_test)
    print("Pruned Test Accuracy:", pruned_accuracy)
    print("Pruned Confusion Matrix:\n", pruned_confusion)
    
    # Plot the pruned decision tree
    tree_model.plot_tree()

"""This code demonstrates an object-oriented approach to building, evaluating, and optimizing a decision tree classifier with pre- and post-pruning techniques. Using Python’s `scikit-learn` library, the `DecisionTreePruning` class encapsulates the decision tree model, allowing for the application of both pre-pruning and post-pruning to manage the complexity of the tree and prevent overfitting. Pre-pruning is implemented through parameters like maximum depth, minimum samples per split, and minimum samples per leaf, which limit the growth of the tree during training. Post-pruning, on the other hand, is handled using cost complexity pruning, where a sequence of trees is generated with different alpha values (complexity parameters) to find the optimal balance between tree complexity and accuracy. The code evaluates the tree’s performance through accuracy and confusion matrix on test data, and includes visualization of both the initial and pruned trees to illustrate the structure and depth of the model before and after pruning. This structured approach provides a practical method for managing decision tree complexity, ensuring a balance between interpretability and predictive performance."""
