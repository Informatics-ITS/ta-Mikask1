import csv
import ast

def load_csv(file_path):
    data = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            data[row['nama']] = row
    return data

def extract_real_filename(file_name_str):
    """Extract actual filename from the string representation of the file list."""
    if not file_name_str or file_name_str == '[]':
        return ""
    
    try:
        file_list = ast.literal_eval(file_name_str)
        if isinstance(file_list, list) and file_list:
            return file_list[0]  # Return the first filename
        return ""
    except (ValueError, SyntaxError):
        # If it's not a list representation, return as is
        return file_name_str

def compare_records():
    # Load both CSV files
    peraturan_data = load_csv('peraturan_metadata.csv')
    updated_data = load_csv('updated_data.csv')
    
    # Initialize categories
    deleted = []
    added = []
    updated = []
    
    # Find deleted records (in peraturan but not in updated)
    for name in peraturan_data:
        if name not in updated_data and name != "Kitab Undang-Undang Hukum Perdata":
            deleted.append({
                'name': name,
                'file_name': peraturan_data[name]['file_name'],
                'description': peraturan_data[name]['description']
            })
    
    # Find added and updated records
    for name in updated_data:
        if name not in peraturan_data:
            # ADD: Name in updated_data but not in peraturan_metadata
            file_name = updated_data[name]['file_name']
            real_file_name = updated_data[name]['real_file_name']
            added.append({
                'name': name,
                'file_name': file_name,
                'real_file_name': real_file_name,
                'description': updated_data[name]['description']
            })
        else:
            # Check if there are unseen file_names to add
            old_files = peraturan_data[name]['file_name']
            new_files = updated_data[name]['file_name']
            
            # Convert string representations to actual lists
            try:
                old_files_list = ast.literal_eval(old_files) if old_files.strip() else []
            except (ValueError, SyntaxError):
                old_files_list = [old_files] if old_files.strip() else []
                
            try:
                new_files_list = ast.literal_eval(new_files) if new_files.strip() else []
            except (ValueError, SyntaxError):
                new_files_list = [new_files] if new_files.strip() else []
            
            # If new files list has files not in old files list
            unseen_files = [f for f in new_files_list if f not in old_files_list and f]
            if unseen_files:
                # Extract real file names for unseen files
                real_unseen_files = updated_data[name]['real_file_name']  # In this case they're already the real filenames
                
                updated.append({
                    'name': name,
                    'old_file_name': old_files,
                    'unseen_file_name': unseen_files,
                    'real_file_name': real_unseen_files,
                    'file_name': old_files_list + unseen_files,
                    'description': peraturan_data[name]['description']
                })
    
    # Save DELETED records to CSV
    with open('DELETED.csv', 'w', newline='', encoding='utf-8') as file:
        fieldnames = ['name', 'file_name', 'description']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for item in deleted:
            writer.writerow(item)
    
    # Save ADDED records to CSV
    with open('ADDED.csv', 'w', newline='', encoding='utf-8') as file:
        fieldnames = ['name', 'file_name', 'real_file_name', 'description']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for item in added:
            writer.writerow(item)
    
    # Save UPDATED records to CSV
    with open('UPDATED.csv', 'w', newline='', encoding='utf-8') as file:
        fieldnames = ['name', 'old_file_name', 'unseen_file_name', 'real_file_name', 'file_name', 'description']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for item in updated:
            writer.writerow(item)
    
    # Print results
    print("=== DELETED RECORDS ===")
    print(f"Total: {len(deleted)} records saved to DELETED.csv")
    
    print("\n=== ADDED RECORDS ===")
    print(f"Total: {len(added)} records saved to ADDED.csv")
    
    print("\n=== UPDATED RECORDS ===")
    print(f"Total: {len(updated)} records saved to UPDATED.csv")
    
    # Update peraturan_metadata with changes
    # For added records
    for item in added:
        peraturan_data[item['name']] = {
            'nama': item['name'],
            'file_name': item['file_name'],
            'description': item['description']
        }
    
    # For updated records - append unseen files
    for item in updated:
        peraturan_data[item['name']]['file_name'] = str(item['file_name'])
    
    # Write updated peraturan_metadata back to CSV
    with open('peraturan_metadata_updated.csv', 'w', newline='', encoding='utf-8') as file:
        fieldnames = ['file_name', 'nama', 'description']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for name in peraturan_data:
            if name not in [item['name'] for item in deleted]:  # Skip deleted records
                writer.writerow(peraturan_data[name])
    
    # Print summary counts
    print("\n=== SUMMARY ===")
    print(f"DELETED: {len(deleted)} records")
    print(f"ADDED: {len(added)} records")
    print(f"UPDATED: {len(updated)} records")
    print(f"Updated peraturan_metadata saved to peraturan_metadata_updated.csv")

if __name__ == "__main__":
    compare_records()
