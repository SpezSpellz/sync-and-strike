
using System;
using System.Collections.Generic;

public class IndexSet<T>
{
    private List<T> items = new List<T>();
    private List<int> indexes = new List<int>();
    private List<int> ids = new List<int>();

    public int add(T obj)
    {
        if (this.items.Count < this.indexes.Count)
        {
            this.items.Add(obj);
            return this.ids[this.items.Count - 1];
        }
        int len = this.items.Count;
        this.indexes.Add(len);
        this.ids.Add(len);
        this.items.Add(obj);
        return len;
    }

    public T remove(int id)
    {
        int len = this.items.Count;
        if (len <= 0)
            throw new IndexOutOfRangeException();
        int idx = this.indexes[id];
        int idx_id = this.ids[idx];
        int last_id = this.ids[len - 1];
        T obj = this.items[idx];
        this.ids[len - 1] = idx_id;
        this.ids[idx] = last_id;
        this.items[idx] = this.items[len - 1];
        this.indexes[last_id] = idx;
        this.indexes[idx_id] = len - 1;
        this.items.RemoveAt(len - 1);
        return obj;
    }

    public T get(int id)
    {
        return this.items[this.indexes[id]];
    }

    public List<T> getList()
    {
        return this.items;
    }
}