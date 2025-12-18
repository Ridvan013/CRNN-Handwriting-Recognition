from database import Database
from werkzeug.security import generate_password_hash
import sys

print("Configuring admin credentials...")
try:
    db = Database()
    conn = db.get_connection()
    cursor = conn.cursor()

    new_hash = generate_password_hash('teacher123')
    cursor.execute("UPDATE users SET password_hash = ? WHERE username = 'admin'", (new_hash,))
    updated = cursor.rowcount > 0
    conn.commit()
    conn.close()

    if updated:
        print("✅ Admin password successfully updated to 'teacher123'")
    else:
        print("⚠️ Admin user not found. Creating new...")
        db.create_user('admin', 'teacher123', 'Administrator', 'admin@ocr.com', 'admin')
        print("✅ Admin user created with password 'teacher123'")

except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)
