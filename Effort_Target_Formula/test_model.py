import pandas as pd
import numpy as np
import pickle
import os

MODEL_PATH = "effort_model.pkl"
DATA_PATH = "test_set_unseen.csv"

def load_model():
    if not os.path.exists(MODEL_PATH):
        print("Erreur: Le modèle n'existe pas. Veuillez lancer 'python train_model.py' en premier.")
        return None, None
    with open(MODEL_PATH, 'rb') as f:
        data = pickle.load(f)
    return data['model'], data['features']

def main():
    print("="*50)
    print("🤖 TESTEUR DE L'ESTIMATEUR D'EFFORT IA")
    print("="*50)
    
    model, features = load_model()
    if model is None:
        return
        
    print("Chargement du jeu de test (données jamais vues par l'IA)...")
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"Erreur: Fichier {DATA_PATH} introuvable. Relancez train_model.py d'abord.")
        return
        
    df = df.fillna({'language': 'Unknown', 'avg_file_size_loc': 0.0, 'comment_per_pr_avg': 0.0})
    
    while True:
        print("\n" + "-"*50)
        user_input = input("Appuyez sur 'Entrée' pour tirer au sort un vrai projet GitHub (ou tapez 'q' pour quitter) : ")
        if user_input.lower() == 'q':
            break
            
        # Tirage au sort d'un projet
        random_index = np.random.randint(0, len(df))
        project = df.iloc[random_index]
        repo_name = project.get('full_name', f'Projet Inconnu #{random_index}')
        
        print(f"\n📁 Projet sélectionné : {repo_name}")
        print(f"⭐ Étoiles : {project['stars']} | 👥 Devs : {project['active_contributors']} | 👨‍💻 Langage : {project['language']}")
        print(f"📊 Fiabilité (Reliability) : {project['reliability_score']}/100")
        
        # Préparation des features pour le modèle
        # Création d'un dataframe avec une seule ligne
        proj_df = pd.DataFrame([project])
        proj_df = pd.get_dummies(proj_df, columns=['language'])
        
        # S'assurer que le projet a bien toutes les colonnes attendues par le modèle (One-Hot Encoding)
        # Si une colonne manque (ex: language_Python n'était pas dans ce projet), on l'ajoute avec 0
        X_test = pd.DataFrame(columns=features)
        for col in features:
            if col in proj_df.columns:
                X_test.loc[0, col] = proj_df[col].values[0]
            else:
                X_test.loc[0, col] = 0
                
        # Convertir en float pour éviter l'erreur XGBoost 'object dtype'
        X_test = X_test.astype(float)
        
        # Prédiction (Le modèle prédit sur une échelle logarithmique)
        y_pred_log = model.predict(X_test)
        # Re-transformation en heures (Exponentielle)
        predicted_hours = np.expm1(y_pred_log)[0]
        actual_hours = project['effort_target']
        
        print("\n🔮 PENSÉE DE L'IA...")
        print(f"-> Effort Prédit par l'IA : {predicted_hours:,.0f} heures".replace(',', ' '))
        print(f"-> Effort Réel constaté   : {actual_hours:,.0f} heures".replace(',', ' '))
        
        error = abs(predicted_hours - actual_hours)
        pct_error = (error / actual_hours) * 100 if actual_hours > 0 else 0
        
        print(f"-> Marge d'erreur : {error:,.0f} heures ({pct_error:.1f}%)".replace(',', ' '))
        
        if pct_error < 25:
            print("✅ EXCELLENTE PRÉDICTION !")
        elif pct_error < 50:
            print("⚠️ PRÉDICTION CORRECTE (Mais peut mieux faire)")
        else:
            print("❌ MAUVAISE PRÉDICTION (Manque de données ou valeurs extrêmes)")

if __name__ == "__main__":
    main()
