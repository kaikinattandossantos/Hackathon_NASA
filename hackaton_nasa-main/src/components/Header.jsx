import Logo from '../assets/logo.png'; 
import Earth from '../assets/earth.png';

const Header = (props) => {
  
  return (
        <div className='bg-main-degrade mb-5'>
          {/* CABEÇALHO */}
          <header className='flex flex-row justify-between items-center min-w-screen h-[90px] bg-black/5'>
            <img src={Logo} alt="" className='ml-10'/>

            <nav className='text-white'>
              <ul className='flex flex-row gap-20 font-normal text-md'>
                <li className='hover:font-bold hover:shadow-lg transition-all duration-700'> <a href=""> Home </a> </li>
                <li className='hover:font-bold hover:shadow-lg transition-all duration-700'> <a href=""> Features </a> </li>
                <li className='hover:font-bold hover:shadow-lg transition-all duration-700'> <a href=""> Dashboard </a> </li>
                <li className='hover:font-bold hover:shadow-lg transition-all duration-700'> <a href=""> About us </a> </li>
              </ul>
            </nav>

            <div></div>
          </header>

          {/* APRESENTAÇÃO GERAL */}
          <div className='flex flex-row'>
            <article className='flex flex-col justify-center w-1/3 h-[450px] ml-24 pt-2 mr-24'>
                <h2 className='text-[50px] mb-2 font-bold leading-tight text-white'>Plan your city's resilient future with <span className='text-our-purple'> NASA data </span></h2>
                <p className='text-lg font-normal text-white'>A platform for city governments and environmental agencies to prioritize science-based investments — heat, air/water, green spaces, and floods. </p>
                <div className='flex flex-row gap-2'>
                  <button className='mt-3 w-1/3 p-4 rounded-lg bg-white hover:bg-white/60 hover:border-2 hover:border-white/90 transition-colors duration-700'>Explore</button>
                  <button className='mt-3 w-1/3 p-4 rounded-lg border-2 border-white text-white hover:bg-white/90 transition-colors duration-700 hover:text-black'>How it works</button>
                </div>
            </article>

            <article className='flex justify-end w-1/3 h-[450px]'>
                <img src={Earth} alt="" className='object-cover hover:scale-110 transition-transform ease-linear duration-500 drop-shadow-lg'/>
            </article>
          </div>
        </div>
  );
};

export default Header;