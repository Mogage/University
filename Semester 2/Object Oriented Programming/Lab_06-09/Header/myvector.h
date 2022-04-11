#pragma once

#define INITIAL_SIZE 1024

template <class TN>
class vector
{
	/// <summary>
	///		Clasa pentru iteratorul unui vector, posibilitatea de 
	///	a face vector<TN>::iterator
	/// </summary>
	class Iterator
	{
	public:
		typedef Iterator	self_type;
		typedef TN			value_type;
		typedef TN&			reference;
		typedef TN*			pointer;
		typedef int			difference_type;
		Iterator(pointer ptr) noexcept : ptr_(ptr) { }
		self_type operator++() noexcept { self_type i = *this; ptr_++; return i; }
		self_type operator++(int junk) noexcept { junk = 0; ptr_++; return *this; }
		reference operator*() noexcept { return *ptr_; }
		pointer operator->() noexcept { return ptr_; }
		bool operator==(const self_type& rhs) noexcept { return ptr_ == rhs.ptr_; }
		bool operator!=(const self_type& rhs) noexcept { return ptr_ != rhs.ptr_; }
	private:
		pointer ptr_;
	};
private:
	TN* Array;
	int Size;
	int Count;

public:
	/// <summary>
	///		Constructor default pentru vector
	/// </summary>
	vector();

	/// <summary>
	///		Constructor pentru un vector cu un anumit size
	/// </summary>
	/// <param name="">Dimensiunea cu care se creeaza</param>
	vector(int);

	/// <summary>
	///		Constructor pentru un vector cu un size si o valoare
	/// </summary>
	/// <param name="">Dimensiunea cu care se creeaza</param>
	/// <param name="">Valoarea initiala pentru vector</param>
	vector(int, TN);

	/// <summary>
	///		Constructor de copiere
	/// </summary>
	/// <param name="">Vectorul care se copiaza</param>
	vector(vector<TN>&);

	/// <summary>
	///		Deconstructor de vector
	/// </summary>
	~vector();

	/// <summary>
	///		Adauga un element la finalul vectorului
	/// </summary>
	/// <param name="">Elementul de adaugat</param>
	void push_back(TN&);

	/// <summary>
	///		Getter pentru numarul de elemente adaugate
	/// </summary>
	/// <returns>Numarul de elemente din vector</returns>
	int size() const noexcept;

	/// <summary>
	///		Sterge un element de pe o anumita pozitie data de iterator
	/// </summary>
	/// <param name="">Pozitia care sa se stearga</param>
	void erase(Iterator);

	/// <summary>
	///		Indica spre prima pozitie din vector
	/// </summary>
	/// <returns>Pozitia de inceput din vector</returns>
	Iterator begin() noexcept { return Iterator(Array); }

	/// <summary>
	///		Indica spre ultima pozitie din vector
	/// </summary>
	/// <returns>Ultima pozitie din vector</returns>
	Iterator end() noexcept { return Iterator(Array + Count); }

	/// <summary>
	///		Suprascrie operatul [] pentru vector
	/// </summary>
	/// <param name="Index">o pozitie din index</param>
	/// <returns>Elementul de pe pozitie</returns>
	TN& operator[](int Index) noexcept;

	/// <summary>
	///		Suprascrie operatorul + intre un vector si int
	/// </summary>
	/// <param name="Number">Numarul cu care se aduna</param>
	/// <returns>Adresa elementului de pe ultima pozitie</returns>
	TN* operator+(int Number);

	/// <summary>
	///		Suprascrie operatorul = dintre 2 vectori
	/// </summary>
	/// <param name=""></param>
	/// <returns>copia vectorului</returns>
	vector<TN>& operator=(const vector<TN>& ToEqual);
};

template <class TN>
vector<TN>::vector()
{
	this->Size = INITIAL_SIZE;
	this->Count = 0;
	this->Array = new TN[INITIAL_SIZE];
}

template <class TN>
vector<TN>::vector(int GivenSize) 
{
	this->Size = GivenSize;
	this->Count = 0;
	this->Array = new TN[GivenSize]();
}

template <class TN>
vector<TN>::vector(int GivenSize, TN InitialValue) 
{
	this->Size = GivenSize;
	this->Count = 0;
	this->Array = new TN[GivenSize];
	for (int index = 0; index < GivenSize; index = index + 1)
	{
		this->Array[index] = InitialValue;
	}
}

template <class TN>
vector<TN>::vector(vector<TN>& CopyVector)
{
	Size = CopyVector.Size;
	Count = CopyVector.Count;
	Array = new TN[Size];

	if (Size > Count)
	{
		for (int i = 0; i < Count; i = i + 1)
		{
			Array[i] = CopyVector.Array[i];
		}
	}
}

template <class TN>
vector<TN>::~vector()
{
	delete[] this->Array;
}

template <class TN>
void vector<TN>::push_back(TN& ToAdd)
{
	if (this->Count < this->Size)
	{
		Array[Count] = ToAdd;
		this->Count = this->Count + 1;
	}
}

template <class TN>
int vector<TN>::size() const noexcept
{
	return this->Count;
}

template<class TN>
inline void vector<TN>::erase(Iterator Pos)
{
	vector<TN>::Iterator it = Array;
	int poz = 0;

	while (it != Pos)
	{
		it++;
		poz = poz + 1;
	}
	for (int index = poz; index < this->Count - 1; index = index + 1)
	{
		this->Array[index] = this->Array[index + 1];
	}
	
	this->Count = this->Count - 1;
}

template<class TN>
inline TN& vector<TN>::operator[](int index) noexcept
{
	return Array[index];
}

template<class TN>
inline TN* vector<TN>::operator+(int Number)
{
	return Array + Number;
}

template<class TN>
inline vector<TN>& vector<TN>::operator=(const vector<TN>& ToEqual)
{
	if (this != &ToEqual && ToEqual.Count < ToEqual.Size)
	{
		TN* newArray = new TN[ToEqual.Size];
		//copiez elementele
		for (int i = 0; i < ToEqual.Count; i++) {
			newArray[i] = ToEqual.Array[i];
		}
		delete[] Array;
		Array = newArray;
		Count = ToEqual.Count;
		Size = ToEqual.Size;
	}

	return *this;
}