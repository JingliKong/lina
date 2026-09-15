module lina;
import std::io;

fn int main(String[] args)
{
	String json_string = `{"name": "Alice", "age": 30}`;
	json::parse(json_string);

	return 0;
}
