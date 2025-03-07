import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split, StratifiedKFold
from deap import base, creator, tools, algorithms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler, StandardScaler
from sklearn.compose import ColumnTransformer

# Random seed
seed = 42
random.seed(seed)
np.random.seed(seed)

# Define the objectives for NSGA-II
creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0, 1.0, 1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti) 

# Define fitness function for Random Forest with detailed fold results
def evalRandomForest(individual, Xx, yy):
    n_estimators, max_depth, min_samples_split, criterion = individual
    num_folds = 5
    n_repeats = 3

    rskf = RepeatedStratifiedKFold(n_splits=num_folds, n_repeats=n_repeats, random_state=seed)

    accuracy_scores = []
    specificity_scores = []
    sensitivity_scores = []
    f1_scores = []

    Xx_np = Xx.to_numpy()
    yy_np = yy.reset_index(drop=True).to_numpy() 

    for train_index, val_index in rskf.split(Xx_np, yy_np):
        X_train_fold, X_val_fold = Xx_np[train_index], Xx_np[val_index]
        y_train_fold, y_val_fold = yy_np[train_index], yy_np[val_index]

        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            criterion=criterion,
            random_state=seed
        )
        clf.fit(X_train_fold, y_train_fold)
        predictions = clf.predict(X_val_fold)

        fold_accuracy_scores = []
        fold_sensitivity_scores = []
        fold_specificity_scores = []
        fold_f1_scores = []

        for c in range(len(set(y_train_fold))):
            tp = np.sum((predictions == c) & (y_val_fold == c))
            tn = np.sum((predictions != c) & (y_val_fold != c))
            fp = np.sum((predictions == c) & (y_val_fold != c))
            fn = np.sum((predictions != c) & (y_val_fold == c))

            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) != 0 else 0
            sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
            f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) != 0 else 0

            fold_accuracy_scores.append(accuracy)
            fold_sensitivity_scores.append(sensitivity)
            fold_specificity_scores.append(specificity)
            fold_f1_scores.append(f1)

        weighted_accuracy = np.min(fold_accuracy_scores)
        weighted_sensitivity = np.min(fold_sensitivity_scores)
        weighted_specificity = np.max(fold_specificity_scores)
        weighted_f1 = np.min(fold_f1_scores)

        accuracy_scores.append(weighted_accuracy)
        sensitivity_scores.append(weighted_sensitivity)
        specificity_scores.append(weighted_specificity)
        f1_scores.append(weighted_f1)

    avg_accuracy = np.mean(accuracy_scores)
    avg_sensitivity = np.mean(sensitivity_scores)
    avg_specificity = np.mean(specificity_scores)
    avg_f1 = np.mean(f1_scores)
    avg_metrics = [avg_accuracy, avg_sensitivity, avg_specificity, avg_f1]
    std_between_metrics = np.std(avg_metrics)
        
    return avg_accuracy, avg_specificity, avg_sensitivity, avg_f1, std_between_metrics

# Create a ParetoFront Hall of Fame instance
hof = tools.ParetoFront()
# Initialize DEAP tools
toolbox = base.Toolbox()
# Parameter ranges
n_estimators_range = np.arange(20, 300)
max_depth_choices = np.arange(5, 31)
min_samples_split_range = np.arange(2, 21)
criterion = ['gini', 'entropy', 'log_loss']

def select_n_estimators():
    return random.choice(n_estimators_range)
def select_max_depth():
    return random.choice(max_depth_choices)
def select_min_samples_split():
    return random.choice(min_samples_split_range)
def select_criterion():
    return random.choice(criterion)

# Register attributes to toolbox
toolbox.register("attr_int_n_estimators", select_n_estimators)
toolbox.register("attr_max_depth", select_max_depth)
toolbox.register("attr_int_min_samples_split", select_min_samples_split)
toolbox.register("attr_criterion", select_criterion)

# Individual and population setup
toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_int_n_estimators, toolbox.attr_max_depth, toolbox.attr_int_min_samples_split, toolbox.attr_criterion), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Custom mutation function to handle the mixed types of parameters
def mutate_n_estimators(individual, indpb):
    if random.random() < indpb:
        individual[0] = select_n_estimators()
    return individual
def mutate_max_depth(individual, indpb):
    if random.random() < indpb:
        individual[1] = select_max_depth()
    return individual
def mutate_min_samples_split(individual, indpb):
    if random.random() < indpb:
        individual[2] = select_min_samples_split()
    return individual
def mutate_criterion(individual, indpb):
    if random.random() < indpb:
        individual[3] = select_criterion()
    return individual

def customMutate(individual, indpb):
    individual = mutate_n_estimators(individual, indpb)
    individual = mutate_max_depth(individual, indpb)
    individual = mutate_min_samples_split(individual, indpb)
    individual = mutate_criterion(individual, indpb)
    return individual

toolbox.register("mutate", customMutate)
toolbox.register("evaluate", evalRandomForest)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("select", tools.selNSGA2)

# Function to save single generation results
def save_generation_results(file_path, gen_data, headers=None):
    with open(file_path, 'a', newline='') as file:
        writer = csv.writer(file)
        if headers:
            writer.writerow(headers)
        writer.writerow(gen_data)
    print(f"Saved generation data to {file_path}")

def calculate_auc_for_individual(individual, X_train_scaled, y_train, X_test_scaled, y_test):
    n_estimators, max_depth, min_samples_split, criterion = individual
    auc_score_list = []
    for c in range(len(set(y_train))):
        y_test_b = y_test == c
        y_train_b = y_train == c
        clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, criterion=criterion, random_state=seed)
        clf.fit(X_train_scaled, y_train_b)
        y_pred_proba = clf.predict_proba(X_test_scaled)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test_b, y_pred_proba)
        auc_score = auc(fpr, tpr)
        auc_score_list.append(auc_score)
    return auc_score_list
from sklearn.ensemble import AdaBoostClassifier, StackingClassifier

def calculate_auc_for_individual_Adaboost (individuals, X_train_scaled, y_train, X_test_scaled, y_test):
    # Fit the StackingClassifier on the training data
    auc_score_list = []
    for c in range(len(set(y_train))):

        y_test_b = y_test == c
        y_train_b = y_train == c
        base_estimators = []
        for i, ind in enumerate(individuals):
            rf = RandomForestClassifier(
                n_estimators=ind[0],
                max_depth=ind[1],
                min_samples_split=ind[2],
                criterion=ind[3],
                random_state=seed
            )
            rf.fit(X_train_scaled, y_train_b)
            base_estimators.append((f"rf_{i}", rf))
                # Create a StackingClassifier with AdaBoost as the final estimator
        stacker = StackingClassifier(
            estimators=base_estimators,
            final_estimator=AdaBoostClassifier(random_state=seed, algorithm= 'SAMME'),
            passthrough=True
        )
        stacker.fit(X_train_scaled, y_train_b)
        y_pred_proba = stacker.predict_proba(X_test_scaled)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test_b, y_pred_proba)
        auc_score = auc(fpr, tpr)
        auc_score_list.append(round(auc_score, 4))
                        # Plot the ROC curve for this class
        plt.figure()  # Create a new figure for each class
        plt.plot(fpr, tpr, 'b', label=f'Class {c} AUC = {auc_score:.4f}')
        plt.plot([0, 1], [0, 1], 'r--')  # Plot the diagonal line (no-skill line)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve for Class {c}')
        plt.legend(loc='lower right')
        
        # Save the ROC curve plot for this class
        plt.savefig(f'./ROC_Curve_NSGAII_AdaBoost_class_{c}.png')
        plt.close()
    return auc_score_list

# Define the main function to run the algorithm
def main():

    file_path = './results.csv'

# Load and preprocess the dataset
    data = pd.read_csv('./heart_failure_clinical_records_dataset.csv')
    data = data.sample(frac=1).reset_index(drop=True)


    # Split the dataset into training and testing sets
    X = data.drop(columns=['DEATH_EVENT'])
    y = data['DEATH_EVENT']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Initialize MinMaxScaler
    scaler = StandardScaler()

    # Fit and transform the training data, and transform the testing data
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert to DataFrame if needed, to maintain the column names
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)

    toolbox.register("evaluate", lambda individual: evalRandomForest(individual, X_train_scaled, y_train))
    
    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        generation_headers = ["Generation", "Evals", "Avg", "Std", "Min", "Max"]
        writer.writerow(generation_headers)
    
    # Create a population and the NSGA-II components
    n_pop = 50
    pop = toolbox.population(n=n_pop)
    hof = tools.ParetoFront(similar=np.array_equal)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    CXPB, MUTPB, NGEN = 0.95, 0.05, 100

    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + stats.fields

    for ind in pop:
        ind.fitness.values = toolbox.evaluate(ind)
        print("Initial evaluation done for an individual.")

    for gen in range(NGEN):
        print("Generation:", gen)
        offspring = toolbox.select(pop, n_pop)
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            toolbox.mutate(mutant, MUTPB)
            del mutant.fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        print("Length of invalid_ind:", len(invalid_ind))

        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
            print("Evaluation done for an individual.", ind.fitness.values)

        pop = toolbox.select(pop + offspring, n_pop)
        for ind in pop:
            print("Individual fitness values:", ind.fitness.values)

        hof.update(pop)

        record = stats.compile(pop)
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        save_generation_results(file_path, [gen, len(invalid_ind), *record['avg'], *record['std'], record['min'], record['max']], generation_headers)

    first_front = tools.sortNondominated(pop, n_pop, first_front_only=True)[0]
    hof.update(first_front)

    print("Hall of Fame:")
    for ind in hof:
        print(ind.fitness.values)

    print(logbook)

    from sklearn.ensemble import AdaBoostClassifier, StackingClassifier

    # base estimators for the ensemble model
    base_estimators = []
    for i, ind in enumerate(hof):
        rf = RandomForestClassifier(
                n_estimators=ind[0],
                max_depth=ind[1],
                min_samples_split=ind[2],
                criterion=ind[3],
                random_state=seed
            )
        rf.fit(X_train_scaled, y_train)
        base_estimators.append((f"rf_{i}", rf))
              
    # Create a StackingClassifier with AdaBoost as the final estimator
    stacker = StackingClassifier(
        estimators=base_estimators,
        final_estimator=AdaBoostClassifier(random_state=seed, algorithm= 'SAMME'),
        passthrough=True
    )

    # Fit the StackingClassifier on the training data
    stacker.fit(X_train_scaled, y_train)

    # Predict on the test data
    y_pred = stacker.predict(X_test_scaled)

    # Evaluate the predictions
    accuracy_scores = []
    specificity_scores = []
    sensitivity_scores = []
    f1_scores = []

    for c in range(len(set(y_train))):
        tp = np.sum((y_pred == c) & (y_test == c))
        tn = np.sum((y_pred != c) & (y_test != c))
        fp = np.sum((y_pred == c) & (y_test != c))
        fn = np.sum((y_pred != c) & (y_test == c))

        accuracy = (tp + tn) / (tp + tn + fp + fn)
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        f1 = 2 * tp / (2 * tp + fp + fn)

        accuracy_scores.append(accuracy)
        sensitivity_scores.append(sensitivity)
        specificity_scores.append(specificity)
        f1_scores.append(f1)

    weights = [np.sum(y_test == c) for c in range(len(set(y_test)))]
    weights = np.array(weights)

    final_accuracy = np.average(accuracy_scores, weights=weights)
    final_sensitivity = np.average(sensitivity_scores, weights=weights)
    final_specificity = np.average(specificity_scores, weights=weights)
    final_f1_score = np.average(f1_scores, weights=weights)
    final_std = np.std([final_accuracy, final_sensitivity, final_specificity, final_f1_score])

    print(f"Final Performance: Accuracy = {final_accuracy:.4f}, Specificity = {final_specificity:.4f}, Sensitivity = {final_sensitivity:.4f}, F1 Score = {final_f1_score:.4f}, Std = {final_std:.4f}")

    # Append the ensemble model's performance to the CSV file
    ensemble_headers = ["Model", "Accuracy", "Specificity", "Sensitivity", "F1 Score", "Std"]

    def append_ensemble_performance(file_path, accuracy, specificity, sensitivity, f1):
        with open(file_path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(ensemble_headers)
            writer.writerow(["Ensemble Model", accuracy, specificity, sensitivity, f1])
        print(f"Appended ensemble model performance to {file_path}")

    append_ensemble_performance(file_path, final_accuracy, final_specificity, final_sensitivity, final_f1_score)
 
    print("Optimal Solutions (Pareto Front):")
    # Print the set of Pareto-optimal solutions after running NSGA-II
    for i, individual in enumerate(hof):
        print(f"\nOptimal Solution {i + 1}:")
        print(f"Parameters: n_estimators={individual[0]}, max_depth={individual[1]}, min_samples_split={individual[2]}, criterion={individual[3]}")
        print(f"Fitness (Accuracy, Specificity, Sensitivity, F1 Score, std): {individual.fitness.values}")

    # After the loop, calculate and print the AUC for the ensemble
    auc_score = calculate_auc_for_individual_Adaboost(hof, X_train_scaled, y_train, X_test_scaled, y_test)
    auc_score = np.average(auc_score, weights=weights)
    print(f"AUC Score Test data for Ensemble Model: {auc_score:.4f}")

    pareto_front_plot = './Pareto_Front_AdaBoost.png'
    pareto_front_objectives = np.array([ind.fitness.values for ind in hof])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pareto_front_objectives[:, 0], pareto_front_objectives[:, 1], pareto_front_objectives[:, 2])
    ax.set_xlabel('Accuracy')
    ax.set_ylabel('Specificity')
    ax.set_zlabel('Sensitivity')
    ax.set_title('Pareto Front for RF optimized with NSGA-II')
    plt.savefig(pareto_front_plot)
    plt.close(fig)
    print(f"Pareto front plot saved to {pareto_front_plot}")

if __name__ == "__main__":
    main()
