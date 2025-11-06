from typing import Dict, Tuple, Any, List
import numpy as np
import pandas as pd
import joblib
import logging
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import VotingClassifier, StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import lightgbm as lgb
import spacy
from src.config import MODEL_CONFIG
from src.utils.data_utils import extract_text_features

logger = logging.getLogger(__name__)

class FakeNewsClassifier:
    """Advanced Fake News Detection model with enhanced feature engineering and model architecture."""
    
    def __init__(self):
        """Initialize the classifier with NLP pipeline and model configurations."""
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except OSError:
            logger.warning("Downloading spaCy model...")
            spacy.cli.download('en_core_web_sm')
            self.nlp = spacy.load('en_core_web_sm')
        
        self.models = {}
        self.best_model = None
        self.feature_pipeline = None
        self.config = MODEL_CONFIG
        
    def create_feature_pipeline(self) -> None:
        """Create advanced feature engineering pipeline."""
        tfidf_config = self.config['feature_engineering']['tfidf']
        
        text_features = Pipeline([
            ('tfidf', TfidfVectorizer(
                ngram_range=tfidf_config['ngram_range'],
                max_features=tfidf_config['max_features'],
                min_df=tfidf_config['min_df'],
                max_df=tfidf_config['max_df'],
                use_idf=True,
                smooth_idf=True,
                sublinear_tf=True
            ))
        ])

        self.feature_pipeline = text_features
        
    def create_models(self) -> None:
        """Create ensemble models including voting and stacking classifiers."""
        lr_params = self.config['model_params']['logistic_regression']
        rf_params = self.config['model_params']['random_forest']
        lgb_params = self.config['model_params']['lightgbm']
        
        # Base models
        lr = LogisticRegression(**lr_params)
        rf = RandomForestClassifier(**rf_params)
        lgb_model = lgb.LGBMClassifier(**lgb_params)
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

        # Stacking Classifier
        # Use a small default CV for stacking to avoid errors on tiny datasets
        stacking_cv = 2
        stacking_clf = StackingClassifier(
            estimators=[
                ('rf', rf),
                ('lgb', lgb_model),
                ('svm', svm)
            ],
            final_estimator=lr,
            cv=stacking_cv
        )

        self.models = {
            'logistic_regression': lr,
            'random_forest': rf,
            'lightgbm': lgb_model,
            'svm': svm,
            'voting': voting_clf,
            'stacking': stacking_clf
        }
        
    def extract_nlp_features(self, text: str) -> Dict[str, Any]:
        """Extract advanced NLP features using spaCy."""
        doc = self.nlp(text)
        
        return {
            'num_entities': len(doc.ents),
            'entity_types': [ent.label_ for ent in doc.ents],
            'text_length': len(text),
            'avg_word_length': np.mean([len(token.text) for token in doc]),
            'num_sentences': len(list(doc.sents)),
            'pos_tags': [token.pos_ for token in doc],
            'dependencies': [token.dep_ for token in doc],
            **extract_text_features(text)
        }

    def train_evaluate_models(self, X_train: pd.Series, y_train: pd.Series,
                            X_val: pd.Series, y_val: pd.Series) -> Dict[str, float]:
        """Train and evaluate all models with cross-validation."""
        from sklearn.model_selection import cross_val_score, StratifiedKFold
        from sklearn.metrics import roc_auc_score
        
        if self.feature_pipeline is None:
            self.create_feature_pipeline()

        if not self.models:
            self.create_models()

        # Transform features
        logger.info("Transforming features...")
        X_train_features = self.feature_pipeline.fit_transform(X_train)
        X_val_features = self.feature_pipeline.transform(X_val)

        results = {}
        best_score = 0
        trained_any = False
        trained_models = []
        
        for name, model in self.models.items():
            logger.info(f"Training and evaluating {name}...")
            # Dynamically choose number of CV splits based on available samples
            n_samples = len(y_train)
            configured_splits = int(self.config['training'].get('n_splits', 5))
            # Ensure at least 2 splits and at most n_samples
            n_splits = max(2, min(configured_splits, n_samples))

            # If there are too few samples for stratification (e.g. a class has
            # only 1 example), StratifiedKFold will fail. Check min class count
            # and fall back to KFold when necessary.
            try:
                min_class_count = int(pd.Series(y_train).value_counts().min())
            except Exception:
                min_class_count = 0

            if min_class_count >= 2 and n_splits <= min_class_count:
                cv = StratifiedKFold(
                    n_splits=n_splits,
                    shuffle=True,
                    random_state=self.config['training']['random_state']
                )
            else:
                # Fall back to plain KFold when stratification is not possible
                from sklearn.model_selection import KFold

                cv = KFold(n_splits=n_splits, shuffle=True, random_state=self.config['training']['random_state'])

            # Cross-validation (guard against small datasets / errors)
            try:
                cv_scores = cross_val_score(
                    model, X_train_features, y_train,
                    cv=cv,
                    scoring='roc_auc'
                )
            except Exception as exc:
                logger.warning(
                    f"Cross-validation failed for {name} (dataset may be too small or invalid labels): {exc}"
                )
                # Use a NaN placeholder so downstream code still has keys
                import numpy as _np

                cv_scores = _np.array([_np.nan])
            
            # Train on full training set (guard against failures on tiny datasets)
            try:
                model.fit(X_train_features, y_train)
                trained_any = True
                trained_models.append(name)
            except Exception as exc:
                logger.warning(f"Training failed for {name}: {exc}")
                # Record NaN scores and continue
                import numpy as _np
                results[name] = {
                    'cv_mean': float(_np.nan),
                    'cv_std': float(_np.nan),
                    'val_score': float(_np.nan)
                }
                continue
            
            # Get validation score
            try:
                try:
                    val_proba = model.predict_proba(X_val_features)[:, 1]
                except AttributeError:
                    # For models that don't support predict_proba
                    val_proba = model.decision_function(X_val_features)

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
            if val_score > best_score:
                best_score = val_score
                self.best_model = model
            
            logger.info(f"{name} - CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
            logger.info(f"{name} - Validation ROC-AUC: {val_score:.4f}")
            
        return results

    def detailed_evaluation(self, X_test: pd.Series, y_test: pd.Series) -> Dict[str, Any]:
        """Perform detailed evaluation of the best model."""
        from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
        
        if self.best_model is None:
            raise ValueError("No model has been trained yet!")
            
        X_test_features = self.feature_pipeline.transform(X_test)
        y_pred = self.best_model.predict(X_test_features)
        
        try:
            y_pred_proba = self.best_model.predict_proba(X_test_features)[:, 1]
        except AttributeError:
            y_pred_proba = self.best_model.decision_function(X_test_features)

        return {
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }

    def save_model(self, filepath: str) -> None:
        """Save the best model and feature pipeline."""
        if self.best_model is None:
            raise ValueError("No model has been trained yet!")
            
        model_data = {
            'model': self.best_model,
            'feature_pipeline': self.feature_pipeline,
            'config': self.config
        }
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")

    @classmethod
    def load_model(cls, filepath: str) -> 'FakeNewsClassifier':
        """Load a saved model."""
        model_data = joblib.load(filepath)
        classifier = cls()
        classifier.best_model = model_data['model']
        classifier.feature_pipeline = model_data['feature_pipeline']
        classifier.config = model_data.get('config', MODEL_CONFIG)
        return classifier

    def predict(self, text: str) -> Tuple[int, float]:
        """Make a prediction for a single text input."""
        if self.best_model is None:
            raise ValueError("Model has not been trained yet!")
            
        # Extract features
        features = self.feature_pipeline.transform([text])
        
        # Make prediction
        prediction = self.best_model.predict(features)[0]
        
        try:
            probability = self.best_model.predict_proba(features)[0][1]
        except AttributeError:
            # For models that don't support predict_proba
            probability = self.best_model.decision_function(features)[0]
            probability = 1 / (1 + np.exp(-probability))  # Convert to probability using sigmoid
        
        return prediction, probability

    def predict_batch(self, texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions for a batch of texts."""
        if self.best_model is None:
            raise ValueError("Model has not been trained yet!")
            
        features = self.feature_pipeline.transform(texts)
        predictions = self.best_model.predict(features)
        
        try:
            probabilities = self.best_model.predict_proba(features)[:, 1]
        except AttributeError:
            probabilities = self.best_model.decision_function(features)
            probabilities = 1 / (1 + np.exp(-probabilities))
            
        return predictions, probabilities