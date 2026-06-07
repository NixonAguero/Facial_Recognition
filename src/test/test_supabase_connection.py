"ESTE SCRIPT PRUEBA LA CONEXIÓN A SUPABASE, REALIZANDO UNA CONSULTA SIMPLE A LA TABLA 'profiles' PARA VERIFICAR QUE LOS DATOS SE PUEDEN RECUPERAR CORRECTAMENTE."

from utils.supabase_client import supabase

response = supabase.table("profiles").select("*").limit(1).execute()
print(response.data)
print("Supabase connection test successful!")