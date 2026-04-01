import flet as ft
import numpy as np
from PIL import Image, ImageDraw
import joblib
import os
from urllib.parse import urlparse
import base64
import io  

os.environ["FLET_SECRET_KEY"] = "super_cle_secrete_ph_123"

if not os.path.exists("uploads"):
    os.makedirs("uploads")

print("Chargement des IA sur le serveur...")
modele_knn = joblib.load('modele_knn_ph.pkl')
modele_vectors = joblib.load('modele_vectors_ph.pkl')
modele_RFR = joblib.load('modele_RFR_ph.pkl')
modele_DTR = joblib.load('modele_DTR_ph.pkl')

def trouver_centre(image_pil):
    img_array = np.array(image_pil.convert('RGB'), dtype=np.int16)
    diff = img_array.max(axis=-1) - img_array.min(axis=-1)
    y_coords, x_coords = np.where(diff > 100)
    
    if len(x_coords) > 0 and len(y_coords) > 0:
        centre_x = int((x_coords.min() + x_coords.max()) / 2)
        centre_y = int((y_coords.min() + y_coords.max()) / 2)
    else:
        l, h = image_pil.size
        centre_x, centre_y = l // 2, h // 2
    return centre_x, centre_y

def recadrer_image(image_pil, centre_x, centre_y): 
    return image_pil.convert('RGB').crop((centre_x - 100, centre_y - 150, centre_x + 100, centre_y + 150))
    
def calculer_ph(image_decoupee):
    image_numpy = np.array(image_decoupee)
    moyenne_rgb = image_numpy.mean(axis=(0, 1)).reshape(1, -1)
    
    ph_knn = round(float(modele_knn.predict(moyenne_rgb)[0]), 1)
    ph_vectors = round(float(modele_vectors.predict(moyenne_rgb)[0]), 1)
    ph_RFR = round(float(modele_RFR.predict(moyenne_rgb)[0]), 1)
    ph_DTR = round(float(modele_DTR.predict(moyenne_rgb)[0]), 1)
    
    return ph_knn, ph_vectors, ph_RFR, ph_DTR

def main(page: ft.Page):
    page.title = "pH'ocus"
    page.horizontal_alignment = "center"
    page.theme_mode = "light"
    page.scroll = "auto"

    # Dictionnaire pour garder en memoire le fichier et les coordonnees actuelles
    etat_app = {"chemin_fichier": None, "cx": 0, "cy": 0}

    titre = ft.Image(
        src="icon.png", 
        width=150, 
        height=150,
        fit=ft.ImageFit.CONTAIN
    )

    texte_resultat_knn = ft.Text("pH Modele KNN : En attente...", size=18)
    texte_resultat_vectors = ft.Text("pH Modele Vecteurs : En attente...", size=18)
    texte_resultat_RFR = ft.Text("pH Modele RFR : En attente...", size=18)
    texte_resultat_DTR = ft.Text("pH Modele DTR : En attente...", size=18)
    
    conteneur_image = ft.Container(width=300, height=300, border=ft.border.all(1, "grey"), border_radius=10)

    # Fonction qui deplace le carre, recalcule et met a jour l'affichage
    def decaler_centre(dx, dy):
        if not etat_app["chemin_fichier"]:
            return
            
        # Mise a jour des coordonnees
        etat_app["cx"] += dx
        etat_app["cy"] += dy
        
        original_image = Image.open(etat_app["chemin_fichier"])
        cx, cy = etat_app["cx"], etat_app["cy"]
        
        # Redessiner le rectangle
        original_image_editable = original_image.copy()
        draw = ImageDraw.Draw(original_image_editable)
        coordonnees_boite = (cx - 100, cy - 150, cx + 100, cy + 150)
        draw.rectangle(coordonnees_boite, outline="red", width=5)
        
        buffer_img = io.BytesIO()
        original_image_editable.save(buffer_img, format="PNG")
        code_image = base64.b64encode(buffer_img.getvalue()).decode('utf-8')
        conteneur_image.content.src_base64 = code_image
        
        # Recalculer le pH
        try:
            img_crop = recadrer_image(original_image, cx, cy)
            ph_knn, ph_vectors, ph_RFR, ph_DTR = calculer_ph(img_crop)
            
            texte_resultat_knn.value = f"pH Modele KNN : {ph_knn}"
            texte_resultat_knn.color = "green" if 6 <= ph_knn <= 8 else "red"
            
            texte_resultat_vectors.value = f"pH Modele Vecteurs : {ph_vectors}"
            texte_resultat_vectors.color = "green" if 6 <= ph_vectors <= 8 else "red"

            texte_resultat_RFR.value = f"pH Modele RFR : {ph_RFR}"
            texte_resultat_RFR.color = "green" if 6 <= ph_RFR <= 8 else "red"

            texte_resultat_DTR.value = f"pH Modele DTR : {ph_DTR}"
            texte_resultat_DTR.color = "green" if 6 <= ph_DTR <= 8 else "red"
            
        except Exception as err:
            texte_resultat_knn.value = f"Erreur IA : {err}"
            texte_resultat_knn.color = "red"
            
        page.update()

    # Boutons de deplacement (masques par defaut)
    btn_haut = ft.IconButton(icon=ft.icons.ARROW_UPWARD, on_click=lambda _: decaler_centre(0, -50), visible=False)
    btn_bas = ft.IconButton(icon=ft.icons.ARROW_DOWNWARD, on_click=lambda _: decaler_centre(0, 50), visible=False)
    btn_gauche = ft.IconButton(icon=ft.icons.ARROW_BACK, on_click=lambda _: decaler_centre(-50, 0), visible=False)
    btn_droite = ft.IconButton(icon=ft.icons.ARROW_FORWARD, on_click=lambda _: decaler_centre(50, 0), visible=False)
    
    ligne_controles = ft.Row([btn_gauche, btn_haut, btn_bas, btn_droite], alignment=ft.MainAxisAlignment.CENTER)

    def on_upload(e: ft.FilePickerUploadEvent):
        if e.progress == 1.0: 
            chemin_fichier = os.path.join("uploads", e.file_name)
            original_image = Image.open(chemin_fichier)
            
            # Enregistrement des informations initiales
            etat_app["chemin_fichier"] = chemin_fichier
            etat_app["cx"], etat_app["cy"] = trouver_centre(original_image)
            
            # Affichage des boutons de controle
            btn_haut.visible = True
            btn_bas.visible = True
            btn_gauche.visible = True
            btn_droite.visible = True
            
            texte_resultat_vectors.value = "" 
            texte_resultat_RFR.value = ""
            texte_resultat_DTR.value = ""
            texte_resultat_knn.value = "Analyse en cours..."
            texte_resultat_knn.color = "blue"
            
            # On force la creation initiale du rectangle vide pour que l'UI se mette en place, 
            # puis on appelle la fonction decaler_centre avec (0,0) pour faire le premier calcul sans rien bouger
            conteneur_image.content = ft.Image(src_base64="", width=300, height=300, fit="contain")
            page.update()
            
            decaler_centre(0, 0)

    def on_result(e: ft.FilePickerResultEvent):
        if e.files:
            try:
                lien_original = page.get_upload_url(e.files[0].name, 60)
                from urllib.parse import urlparse
                parsed = urlparse(lien_original)
                lien_relatif = f"{parsed.path}?{parsed.query}"
                
                texte_resultat_knn.value = "Envoi de l'image..."
                texte_resultat_knn.color = "blue"
                
                texte_resultat_vectors.value = ""
                texte_resultat_RFR.value = ""
                texte_resultat_DTR.value = ""
                
                # Masquer les boutons pendant le chargement d'une nouvelle photo
                btn_haut.visible = False
                btn_bas.visible = False
                btn_gauche.visible = False
                btn_droite.visible = False
                page.update()
                
                selecteur.upload([
                    ft.FilePickerUploadFile(e.files[0].name, upload_url=lien_relatif)
                ])
                
            except Exception as err:
                texte_resultat_knn.value = f"Erreur : {err}"
                texte_resultat_knn.color = "red"
                page.update()

    selecteur = ft.FilePicker()
    selecteur.on_result = on_result
    selecteur.on_upload = on_upload
    page.overlay.append(selecteur)

    bouton = ft.ElevatedButton("Prendre une photo", on_click=lambda _: selecteur.pick_files())

    page.add(titre, bouton, conteneur_image, ligne_controles, texte_resultat_knn, texte_resultat_vectors, texte_resultat_RFR, texte_resultat_DTR)

port = int(os.environ.get("PORT", 8000))
ft.app(target=main, host="0.0.0.0", port=port, upload_dir="uploads", assets_dir="assets")
