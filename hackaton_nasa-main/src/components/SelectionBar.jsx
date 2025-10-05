const SelectionBar = (props) => {
  
  return (
    <div className='bg-bar-degrade border-2 rounded-sm border-white/50 mx-2 backdrop-blur-lg'>
            <article className='flex flex-row justify-between items-center w-full h-auto'>

                <form action="" className='flex flex-row p-5 justify-center items-center'>
                      <label htmlFor="city-select" className='text-[16px] mb-1 font-bold mr-2'> Cidade </label>
                      <select name="city" id="city-select" className='appearance-none p-1 text-[14px] font-light rounded-md bg-white/60 border-white border-2 shadow-inner hover:bg-white/50 transition-all ease-in-out'>
                          <option value="" disabled selected> Selecione uma cidade </option>
                          <option value=""></option>
                      </select>
                </form>

                {/* <div>
                  <button className='bg-purple-labubu text-white shadow-inner border-white/70 border-2 p-5 mx-5 rounded-lg font-semibold hover:border-white/80 hover:border-2 hover:bg-white/40 duration-1000'> Assistente virtual </button>
                </div> */}

            </article>
          </div>
  );
};

export default SelectionBar;