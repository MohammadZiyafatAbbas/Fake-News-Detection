import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
import lightgbm as lgb
import joblib
import logging
from typing import Tuple, Dict, Any
import spacy
import en_core_web_sm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedFakeNewsClassifier:
    def __init__(self):
        """Initialize the advanced fake news classifier with enhanced feature engineering and model architecture."""
        self.nlp = en_core_web_sm.load()
        self.models = {}
        self.best_model = None
        self.feature_pipeline = None
        
    def _create_text_features(self, text: str) -> Dict[str, float]:
        """Extract advanced NLP features from text using spaCy."""
        doc = self.nlp(text)
        return {
            'num_entities': len(doc.ents),
            'text_length': len(text),
            'avg_word_length': np.mean([len(token.text) for token in doc]),
            'num_punctuations': sum([1 for token in doc if token.is_punct]),
            'num_capital_words': sum([1 for token in doc if token.text.isupper()]),
            'sentiment': doc.sentiment if hasattr(doc, 'sentiment') else 0.0
        }
    
    def create_feature_pipeline(self) -> None:
        """Create an advanced feature engineering pipeline."""
        text_features = Pipeline([
            ('tfidf', TfidfVectorizer(
                ngram_range=(1, 3),
                max_features=50000,
                min_df=2,
                max_df=0.95,
                use_idf=True,
                smooth_idf=True,
                sublinear_tf=True
            ))
        ])

        self.feature_pipeline = text_features

    def create_models(self) -> None:
        """Create an ensemble of models including voting and stacking classifiers."""
        # Base models
        lr = LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced')
        rf = RandomForestClassifier(n_estimators=200, max_depth=20, class_weight='balanced')
        lgb_model = lgb.LGBMClassifier(n_estimators=200, num_leaves=31, class_weight='balanced')
        svm = LinearSVC(class_weight='balanced', dual=False)

        # Voting Classifier
        voting_clf = VotingClassifier(
            estimators=[
                ('lr', lr),
                ('rf', rf),
                ('lgb', lgb_model)
            ],
            voting='soft'
        )

        # Stacking Classifier - use a small safe CV to avoid errors on tiny datasets
        stacking_clf = StackingClassifier(
            estimators=[
                ('rf', rf),
                ('lgb', lgb_model),
                ('svm', svm)
            ],
            final_estimator=lr,
            cv=2
        )

        self.models = {
            'logistic_regression': lr,
            'random_forest': rf,
            'lightgbm': lgb_model,
            'svm': svm,
            'voting': voting_clf,
            'stacking': stacking_clf
        }

    def train_evaluate_models(self, X_train: pd.Series, y_train: pd.Series,
                            X_val: pd.Series, y_val: pd.Series) -> Dict[str, float]:
        """Train and evaluate all models using cross-validation."""
        if self.feature_pipeline is None:
            self.create_feature_pipeline()

        if not self.models:
            self.create_models()

        # Transform features
        X_train_features = self.feature_pipeline.fit_transform(X_train)
        X_val_features = self.feature_pipeline.transform(X_val)

        results = {}
        best_score = 0
        
        for name, model in self.models.items():
            logger.info(f"Training and evaluating {name}...")

            # Choose CV splits dynamically based on available samples
            n_samples = len(y_train)
            configured_splits = 5
            n_splits = max(2, min(configured_splits, n_samples))

            # Prefer StratifiedKFold when possible
            try:
                min_class_count = int(pd.Series(y_train).value_counts().min())
            except Exception:
                min_class_count = 0

            if min_class_count >= 2 and n_splits <= min_class_count:
                from sklearn.model_selection import StratifiedKFold as _StratifiedKFold
                cv = _StratifiedKFold(n_splits=n_splits, shuffle=True)
            else:
                from sklearn.model_selection import KFold as _KFold
                cv = _KFold(n_splits=n_splits, shuffle=True)

            # Cross-validation (guarded)
            try:
                cv_scores = cross_val_score(
                    model, X_train_features, y_train,
                    cv=cv,
                    scoring='roc_auc'
                )
            except Exception as exc:
                logger.warning(f"Cross-validation failed for {name}: {exc}")
                import numpy as _np
                cv_scores = _np.array([_np.nan])

            # Train on full training set (guarded)
            try:
                model.fit(X_train_features, y_train)
            except Exception as exc:
                logger.warning(f"Training failed for {name}: {exc}")
                import numpy as _np
                results[name] = {
                    'cv_mean': float(_np.nan),
                    'cv_std': float(_np.nan),
                    'val_score': float(_np.nan)
                }
                continue

            try:
                val_proba = model.predict_proba(X_val_features)[:, 1]
            except Exception:
                try:
                    val_proba = model.decision_function(X_val_features)
                except Exception as exc:
                    logger.warning(f"Could not obtain validation probabilities for {name}: {exc}")
                    import numpy as _np
                    val_proba = _np.array([_np.nan])

            try:
                val_score = roc_auc_score(y_val, val_proba)
            except Exception as exc:
                logger.warning(f"Validation scoring failed for {name}: {exc}")
                import numpy as _np
                val_score = float(_np.nan)

            results[name] = {
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'val_score': val_score
            }

            # Track best model
            if val_score == val_score and val_score > best_score:
                best_score = val_score
                self.best_model = model

            logger.info(f"{name} - CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
            logger.info(f"{name} - Validation ROC-AUC: {val_score:.4f}")
            
        return results

    def detailed_evaluation(self, X_test: pd.Series, y_test: pd.Series) -> Dict[str, Any]:
        """Perform detailed evaluation of the best model."""
        X_test_features = self.feature_pipeline.transform(X_test)
        y_pred = self.best_model.predict(X_test_features)
        y_pred_proba = self.best_model.predict_proba(X_test_features)[:, 1]

        return {
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }

    def save_model(self, filepath: str) -> None:
        """Save the best model and feature pipeline."""
        model_data = {
            'model': self.best_model,
            'feature_pipeline': self.feature_pipeline
        }
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")

    @staticmethod
    def load_model(filepath: str) -> 'AdvancedFakeNewsClassifier':
        """Load a saved model."""
        model_data = joblib.load(filepath)
        classifier = AdvancedFakeNewsClassifier()
        classifier.best_model = model_data['model']
        classifier.feature_pipeline = model_data['feature_pipeline']
        return classifier

    def predict(self, text: str) -> Tuple[int, float]:
        """Make a prediction for a single text input."""
        if self.best_model is None:
            raise ValueError("Model has not been trained yet!")
            
        features = self.feature_pipeline.transform([text])
        prediction = self.best_model.predict(features)[0]
        probability = self.best_model.predict_proba(features)[0][1]
        
        return prediction, probability