import sys
import os

# Add backend/ to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from run import get_db_connection, generate_password_hash

def create_admin_user():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    username = "admin"
    password = "password123"
    role = "admin"
    full_name = "Administrator"
    
    hashed_pw = generate_password_hash(password, method='scrypt')
    
    try:
        cursor.execute(
            'INSERT INTO users (username, password_hash, role, full_name) VALUES (%s, %s, %s, %s)',
            (username, hashed_pw, role, full_name)
        )
        conn.commit()
        print(f"User created successfully!")
        print(f"Username: {username}")
        print(f"Password: {password}")
        print(f"Role: {role}")
    except Exception as e:
        if "unique" in str(e).lower() or "already exists" in str(e).lower():
            print(f"User '{username}' already exists. Attempting to update password...")
            try:
                cursor.execute(
                    'UPDATE users SET password_hash = %s, role = %s, full_name = %s WHERE username = %s',
                    (hashed_pw, role, full_name, username)
                )
                conn.commit()
                print(f"User updated successfully!")
                print(f"Username: {username}")
                print(f"Password: {password}")
                print(f"Role: {role}")
            except Exception as update_err:
                print(f"Failed to update user: {update_err}")
        else:
            print(f"Failed to create user: {e}")
    finally:
        cursor.close()
        conn.close()

if __name__ == '__main__':
    create_admin_user()
